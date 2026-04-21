import argparse
import gc
import json
from collections import defaultdict, deque

import numpy as np
import spacy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Syntactic label extraction (gold labels from original/possible sentences)
# ---------------------------------------------------------------------------

class SyntacticLabeler:
    """Extract gold POS tags, dependency relations, and syntactic depth
    from unscrambled (possible) sentences using spaCy."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def extract_labels(self, sentence: str):
        """Return per-token POS, dep relation, head index, and tree depth."""
        doc = self.nlp(sentence)
        tokens, pos_tags, dep_rels, head_indices, depths = [], [], [], [], []
        for tok in doc:
            tokens.append(tok.text)
            pos_tags.append(tok.pos_)          # Universal POS tag
            dep_rels.append(tok.dep_)           # dependency relation label
            head_indices.append(tok.head.i)     # index of syntactic head
            depths.append(self._tree_depth(tok))
        return {
            "tokens": tokens,
            "pos": pos_tags,
            "dep_rel": dep_rels,
            "head_idx": head_indices,
            "depth": depths,
        }

    @staticmethod
    def _tree_depth(token) -> int:
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        return depth


# ---------------------------------------------------------------------------
# Representation extraction from attention heads
# ---------------------------------------------------------------------------

class HeadRepresentationExtractor:
    """Load a GPT-2 model and extract per-head output representations."""

    def __init__(self, model_path: str, tokenizer_path: str = None, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        tok_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            output_hidden_states=False,
        )
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.n_layers = cfg.n_layer
        self.n_heads = cfg.n_head
        self.d_head = cfg.n_embd // cfg.n_head  # 64 for GPT-2
        self.model_path = model_path

        # Register pre-hooks on c_proj to capture per-head outputs BEFORE the
        # output projection merges heads. The input to c_proj is the
        # concatenation [head_0 || head_1 || ... || head_{H-1}] of true
        # per-head outputs; reshaping to (B, seq, n_heads, d_head) recovers
        # each head's genuine d_head-dim output.
        self._head_outputs = {}
        self._hooks = []
        self._register_hooks()

    # -- hook machinery -----------------------------------------------------

    def _register_hooks(self):
        for layer_idx in range(self.n_layers):
            c_proj = self.model.transformer.h[layer_idx].attn.c_proj
            hook = c_proj.register_forward_pre_hook(
                self._make_pre_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_pre_hook(self, layer_idx: int):
        def hook_fn(module, inputs):
            # inputs is a tuple; the first element is the tensor that feeds
            # c_proj, shape (B, seq, n_embd). This tensor is the concatenation
            # of all heads' outputs BEFORE c_proj linearly mixes them.
            pre_proj = inputs[0]
            batch, seq, _ = pre_proj.shape
            per_head = pre_proj.view(batch, seq, self.n_heads, self.d_head)
            self._head_outputs[layer_idx] = per_head.detach().cpu()
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- extraction ---------------------------------------------------------

    def extract(self, text: str):

        self._head_outputs.clear()
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = [tuple(span) for span in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        seq_len = inputs["input_ids"].shape[1]
        if seq_len == 0:
            return None, [], [], []

        with torch.no_grad():
            self.model(**inputs)

        token_ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        head_reps = {}
        for layer_idx in range(self.n_layers):
            per_head = self._head_outputs[layer_idx][0]  # (seq, n_heads, d_head)
            for head_idx in range(self.n_heads):
                arr = per_head[:, head_idx, :].numpy()
                # Replace NaN/Inf at source to avoid downstream data loss
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                head_reps[(layer_idx, head_idx)] = arr

        return head_reps, token_ids, tokens, offsets


# ---------------------------------------------------------------------------
# Subword-to-word alignment (identity-based, as described in probes.tex §3.2)
# ---------------------------------------------------------------------------

def _first_subword_for_char_span(offsets, start_char: int, end_char: int):
    """Return the first subword index whose character span overlaps
    [start_char, end_char)."""
    for sw_idx, (sub_start, sub_end) in enumerate(offsets):
        if sub_end <= start_char:
            continue
        if sub_start >= end_char:
            break
        if max(sub_start, start_char) < min(sub_end, end_char):
            return sw_idx
    return None


def align_scrambled_to_original_by_identity(
    scrambled_sentence: str,
    offsets,
    labeler: "SyntacticLabeler",
    word_labels: dict,
):
    """Map each word of the scrambled sentence to a gold label from the
    original parse **by token identity** (text match), preserving occurrence
    multiplicity with FIFO matching.

    This implements the mapping described in probes.tex §3.2: "dependencies
    mapped to scrambled positions by word identity".

    Returns a list of dicts, one per successfully matched scrambled word,
    containing:
        subword_idx        - index into the GPT-2 subword sequence
        scrambled_word_idx - position of this word in the scrambled sentence
        word_idx           - index of the matched token in the original parse
        token              - token text
        pos, dep_rel, head_idx, depth  - gold labels inherited from original
    """
    scrambled_doc = labeler.nlp.make_doc(scrambled_sentence)

    # Build FIFO pools of original indices keyed by token text.
    original_pool = defaultdict(deque)
    for original_idx, tok_text in enumerate(word_labels["tokens"]):
        original_pool[tok_text].append(original_idx)

    aligned = []
    for scrambled_word_idx, tok in enumerate(scrambled_doc):
        pool = original_pool.get(tok.text)
        if not pool:
            continue
        original_idx = pool.popleft()

        subword_idx = _first_subword_for_char_span(
            offsets, tok.idx, tok.idx + len(tok)
        )
        if subword_idx is None:
            continue

        aligned.append({
            "subword_idx": subword_idx,
            "scrambled_word_idx": scrambled_word_idx,
            "word_idx": original_idx,
            "token": tok.text,
            "pos": word_labels["pos"][original_idx],
            "dep_rel": word_labels["dep_rel"][original_idx],
            "head_idx": word_labels["head_idx"][original_idx],
            "depth": word_labels["depth"][original_idx],
        })

    return aligned


# ---------------------------------------------------------------------------
# Dataset construction for probing
# ---------------------------------------------------------------------------

def build_probing_dataset(
    extractor: HeadRepresentationExtractor,
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    max_sentences: int = None,
):
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]

    n_layers = extractor.n_layers
    n_heads = extractor.n_heads

    dataset = {
        (l, h): {
            "X": [], "pos": [], "dep_rel": [], "depth": [],
            "head_idx": [], "sentence_id": [],
        }
        for l in range(n_layers)
        for h in range(n_heads)
    }

    skipped = 0
    iterator = tqdm(
        zip(scrambled_sentences, original_sentences),
        total=len(scrambled_sentences),
        desc="Probing dataset",
        unit="sent",
    )
    for i, (scrambled, original) in enumerate(iterator):

        # Gold labels from original sentence
        labels = labeler.extract_labels(original)

        # Model representations from scrambled sentence
        head_reps, token_ids, tokens, offsets = extractor.extract(scrambled)
        if head_reps is None:
            skipped += 1
            continue

        # Identity-based alignment of scrambled words to original gold labels
        aligned = align_scrambled_to_original_by_identity(
            scrambled, offsets, labeler, labels,
        )
        if len(aligned) == 0:
            skipped += 1
            continue

        # Collect aligned representations and labels
        for entry in aligned:
            sw_idx = entry["subword_idx"]
            for l in range(n_layers):
                for h in range(n_heads):
                    rep = head_reps[(l, h)]
                    if sw_idx < rep.shape[0]:
                        dataset[(l, h)]["X"].append(rep[sw_idx])
                        dataset[(l, h)]["pos"].append(entry["pos"])
                        dataset[(l, h)]["dep_rel"].append(entry["dep_rel"])
                        dataset[(l, h)]["depth"].append(entry["depth"])
                        dataset[(l, h)]["head_idx"].append(entry["head_idx"])
                        dataset[(l, h)]["sentence_id"].append(i)

    if skipped:
        print(f"  Skipped {skipped}/{len(scrambled_sentences)} sentences (alignment)")

    # Convert lists to arrays
    for key in dataset:
        if len(dataset[key]["X"]) > 0:
            dataset[key]["X"] = np.stack(dataset[key]["X"])
            dataset[key]["depth"] = np.array(dataset[key]["depth"], dtype=np.float32)
            dataset[key]["sentence_id"] = np.array(
                dataset[key]["sentence_id"], dtype=np.int32,
            )
        else:
            dataset[key]["X"] = np.empty((0, extractor.d_head))
            dataset[key]["depth"] = np.empty(0)
            dataset[key]["sentence_id"] = np.empty(0, dtype=np.int32)

    return dataset


# ---------------------------------------------------------------------------
# PyTorch linear probes
# ---------------------------------------------------------------------------

PROBE_BATCH_SIZE = 4096
PROBE_MAX_EPOCHS = 12
PROBE_PATIENCE = 2
PROBE_LR = 1e-2
PROBE_WEIGHT_DECAY = 1e-4
TOKEN_HEAD_CHUNK_SIZE = 24
PAIRWISE_HEAD_CHUNK_SIZE = 1
PAIRWISE_FEATURE_DTYPE = np.float16


def _default_probe_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _standardise_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train_std = ((X_train - mean) / std).astype(np.float32, copy=False)
    X_test_std = ((X_test - mean) / std).astype(np.float32, copy=False)
    return X_train_std, X_test_std


def _make_validation_split(X_train, y_train, random_state, is_classification):
    if len(X_train) < 100:
        return X_train, None, y_train, None

    stratify = None
    if is_classification:
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) > 1 and np.all(counts >= 2):
            stratify = y_train

    try:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return X_train, None, y_train, None

    return X_fit, X_val, y_fit, y_val


def _build_loader(X, y, batch_size, shuffle, random_state, is_classification):
    x_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
    if is_classification:
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))
    else:
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))

    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator().manual_seed(random_state)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate_classifier_loss(model, X_val, y_val, criterion, device):
    if X_val is None or y_val is None or len(X_val) == 0:
        return None

    loader = _build_loader(
        X_val, y_val, batch_size=PROBE_BATCH_SIZE, shuffle=False,
        random_state=0, is_classification=True,
    )
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            logits = model(xb)
            losses.append(float(criterion(logits, yb).detach().cpu()))
    return float(np.mean(losses)) if losses else None


def _evaluate_regression_loss(model, X_val, y_val, criterion, device):
    if X_val is None or y_val is None or len(X_val) == 0:
        return None

    loader = _build_loader(
        X_val, y_val, batch_size=PROBE_BATCH_SIZE, shuffle=False,
        random_state=0, is_classification=False,
    )
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            preds = model(xb).squeeze(-1)
            losses.append(float(criterion(preds, yb).detach().cpu()))
    return float(np.mean(losses)) if losses else None


def _predict_classifier(model, X_test, device):
    loader = _build_loader(
        X_test,
        np.zeros(len(X_test), dtype=np.int64),
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        random_state=0,
        is_classification=True,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.int64)


def _predict_regression(model, X_test, device):
    loader = _build_loader(
        X_test,
        np.zeros(len(X_test), dtype=np.float32),
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        random_state=0,
        is_classification=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            preds.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)


def _fit_linear_classifier_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_classes: int,
    device: torch.device,
    random_state: int,
    class_weights: np.ndarray | None = None,
):
    X_train_std, X_test_std = _standardise_train_test(X_train, X_test)
    X_fit, X_val, y_fit, y_val = _make_validation_split(
        X_train_std, y_train, random_state, is_classification=True,
    )

    torch.manual_seed(random_state)
    model = nn.Linear(X_train_std.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=(
            torch.tensor(class_weights, dtype=torch.float32, device=device)
            if class_weights is not None else None
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY,
    )

    train_loader = _build_loader(
        X_fit, y_fit, batch_size=PROBE_BATCH_SIZE, shuffle=True,
        random_state=random_state, is_classification=True,
    )

    best_state = None
    best_loss = float("inf")
    patience_left = PROBE_PATIENCE

    for epoch in range(PROBE_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        monitor = _evaluate_classifier_loss(model, X_val, y_val, criterion, device)
        if monitor is None:
            monitor = _evaluate_classifier_loss(model, X_fit, y_fit, criterion, device)

        if monitor < best_loss - 1e-4:
            best_loss = monitor
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_left = PROBE_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return _predict_classifier(model, X_test_std, device)


def _fit_linear_regressor_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    device: torch.device,
    random_state: int,
):
    X_train_std, X_test_std = _standardise_train_test(X_train, X_test)
    X_fit, X_val, y_fit, y_val = _make_validation_split(
        X_train_std, y_train, random_state, is_classification=False,
    )

    torch.manual_seed(random_state)
    model = nn.Linear(X_train_std.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY,
    )

    train_loader = _build_loader(
        X_fit, y_fit, batch_size=PROBE_BATCH_SIZE, shuffle=True,
        random_state=random_state, is_classification=False,
    )

    best_state = None
    best_loss = float("inf")
    patience_left = PROBE_PATIENCE

    for epoch in range(PROBE_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        monitor = _evaluate_regression_loss(model, X_val, y_val, criterion, device)
        if monitor is None:
            monitor = _evaluate_regression_loss(model, X_fit, y_fit, criterion, device)

        if monitor < best_loss - 1e-5:
            best_loss = monitor
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_left = PROBE_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return _predict_regression(model, X_test_std, device)


def _balanced_class_weights(y: np.ndarray, n_classes: int):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.ones(n_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = len(y) / (n_classes * counts[nonzero])
    return weights


def _make_validation_indices(y_train, random_state, is_classification):
    if len(y_train) < 100:
        return np.arange(len(y_train)), None

    stratify = None
    if is_classification:
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) > 1 and np.all(counts >= 2):
            stratify = y_train

    indices = np.arange(len(y_train))
    try:
        fit_idx, val_idx = train_test_split(
            indices,
            test_size=0.1,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return indices, None

    return np.asarray(fit_idx), np.asarray(val_idx)


def _standardise_multihead_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=1, keepdims=True)
    std = X_train.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train_std = ((X_train - mean) / std).astype(np.float32, copy=False)
    X_test_std = ((X_test - mean) / std).astype(np.float32, copy=False)
    return X_train_std, X_test_std


def _evaluate_multihead_classifier_loss(W, b, X_eval_t, y_eval_t, weight_t):
    if X_eval_t is None or y_eval_t is None or y_eval_t.numel() == 0:
        return None

    n_heads = X_eval_t.shape[0]
    losses = torch.zeros(n_heads, device=X_eval_t.device)
    total = 0
    for start in range(0, y_eval_t.shape[0], PROBE_BATCH_SIZE):
        stop = min(start + PROBE_BATCH_SIZE, y_eval_t.shape[0])
        xb = X_eval_t[:, start:stop, :]
        yb = y_eval_t[start:stop]
        logits = torch.einsum("hbd,hcd->hbc", xb, W) + b[:, None, :]
        loss = F.cross_entropy(
            logits.permute(0, 2, 1),
            yb.unsqueeze(0).expand(n_heads, -1),
            reduction="none",
            weight=weight_t,
        )
        losses += loss.sum(dim=1)
        total += (stop - start)
    return losses / max(total, 1)


def _predict_multihead_classifier(W, b, X_test_std: np.ndarray, device):
    X_test_t = torch.from_numpy(X_test_std).to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, X_test_t.shape[1], PROBE_BATCH_SIZE):
            stop = min(start + PROBE_BATCH_SIZE, X_test_t.shape[1])
            xb = X_test_t[:, start:stop, :]
            logits = torch.einsum("hbd,hcd->hbc", xb, W) + b[:, None, :]
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds, axis=1) if preds else np.empty((X_test_std.shape[0], 0), dtype=np.int64)


def _fit_multihead_linear_classifier_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_classes: int,
    device: torch.device,
    random_state: int,
    class_weights: np.ndarray | None = None,
):
    X_train_std, X_test_std = _standardise_multihead_train_test(X_train, X_test)
    fit_idx, val_idx = _make_validation_indices(y_train, random_state, is_classification=True)

    X_fit_t = torch.from_numpy(X_train_std[:, fit_idx, :]).to(device)
    y_fit_t = torch.from_numpy(np.asarray(y_train[fit_idx], dtype=np.int64)).to(device)
    if val_idx is not None:
        X_val_t = torch.from_numpy(X_train_std[:, val_idx, :]).to(device)
        y_val_t = torch.from_numpy(np.asarray(y_train[val_idx], dtype=np.int64)).to(device)
    else:
        X_val_t = None
        y_val_t = None

    n_heads = X_train_std.shape[0]
    d_head = X_train_std.shape[2]
    torch.manual_seed(random_state)
    W = nn.Parameter(torch.empty((n_heads, n_classes, d_head), device=device))
    b = nn.Parameter(torch.zeros((n_heads, n_classes), device=device))
    nn.init.xavier_uniform_(W)

    optimizer = torch.optim.AdamW(
        [W, b], lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY,
    )
    weight_t = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None else None
    )

    best_loss = torch.full((n_heads,), float("inf"), device=device)
    best_W = W.detach().cpu().clone()
    best_b = b.detach().cpu().clone()
    patience_left = torch.full((n_heads,), PROBE_PATIENCE, dtype=torch.int64, device=device)
    rng = np.random.default_rng(random_state)

    for _ in range(PROBE_MAX_EPOCHS):
        order = rng.permutation(len(fit_idx))
        for start in range(0, len(order), PROBE_BATCH_SIZE):
            idx = order[start:start + PROBE_BATCH_SIZE]
            xb = X_fit_t[:, idx, :]
            yb = y_fit_t[idx]
            logits = torch.einsum("hbd,hcd->hbc", xb, W) + b[:, None, :]
            loss = F.cross_entropy(
                logits.permute(0, 2, 1),
                yb.unsqueeze(0).expand(n_heads, -1),
                weight=weight_t,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        monitor = _evaluate_multihead_classifier_loss(W, b, X_val_t, y_val_t, weight_t)
        if monitor is None:
            monitor = _evaluate_multihead_classifier_loss(W, b, X_fit_t, y_fit_t, weight_t)

        improved = monitor < (best_loss - 1e-4)
        if improved.any():
            W_cpu = W.detach().cpu()
            b_cpu = b.detach().cpu()
            improved_cpu = improved.detach().cpu()
            best_W[improved_cpu] = W_cpu[improved_cpu]
            best_b[improved_cpu] = b_cpu[improved_cpu]
            best_loss = torch.where(improved, monitor, best_loss)
            patience_left = torch.where(
                improved,
                torch.full_like(patience_left, PROBE_PATIENCE),
                patience_left - 1,
            )
        else:
            patience_left -= 1

        if bool((patience_left <= 0).all()):
            break

    W = best_W.to(device)
    b = best_b.to(device)
    return _predict_multihead_classifier(W, b, X_test_std, device)


def _evaluate_multihead_regression_loss(W, b, X_eval_t, y_eval_t):
    if X_eval_t is None or y_eval_t is None or y_eval_t.numel() == 0:
        return None

    n_heads = X_eval_t.shape[0]
    losses = torch.zeros(n_heads, device=X_eval_t.device)
    total = 0
    for start in range(0, y_eval_t.shape[0], PROBE_BATCH_SIZE):
        stop = min(start + PROBE_BATCH_SIZE, y_eval_t.shape[0])
        xb = X_eval_t[:, start:stop, :]
        yb = y_eval_t[start:stop]
        preds = torch.einsum("hbd,hd->hb", xb, W) + b[:, None]
        loss = (preds - yb.unsqueeze(0).expand(n_heads, -1)) ** 2
        losses += loss.sum(dim=1)
        total += (stop - start)
    return losses / max(total, 1)


def _predict_multihead_regressor(W, b, X_test_std: np.ndarray, device):
    X_test_t = torch.from_numpy(X_test_std).to(device)
    preds = []
    with torch.no_grad():
        for start in range(0, X_test_t.shape[1], PROBE_BATCH_SIZE):
            stop = min(start + PROBE_BATCH_SIZE, X_test_t.shape[1])
            xb = X_test_t[:, start:stop, :]
            preds.append((torch.einsum("hbd,hd->hb", xb, W) + b[:, None]).cpu().numpy())
    return np.concatenate(preds, axis=1) if preds else np.empty((X_test_std.shape[0], 0), dtype=np.float32)


def _fit_multihead_linear_regressor_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    device: torch.device,
    random_state: int,
):
    X_train_std, X_test_std = _standardise_multihead_train_test(X_train, X_test)
    fit_idx, val_idx = _make_validation_indices(y_train, random_state, is_classification=False)

    X_fit_t = torch.from_numpy(X_train_std[:, fit_idx, :]).to(device)
    y_fit_t = torch.from_numpy(np.asarray(y_train[fit_idx], dtype=np.float32)).to(device)
    if val_idx is not None:
        X_val_t = torch.from_numpy(X_train_std[:, val_idx, :]).to(device)
        y_val_t = torch.from_numpy(np.asarray(y_train[val_idx], dtype=np.float32)).to(device)
    else:
        X_val_t = None
        y_val_t = None

    n_heads = X_train_std.shape[0]
    d_head = X_train_std.shape[2]
    torch.manual_seed(random_state)
    W = nn.Parameter(torch.empty((n_heads, d_head), device=device))
    b = nn.Parameter(torch.zeros((n_heads,), device=device))
    nn.init.xavier_uniform_(W)

    optimizer = torch.optim.AdamW(
        [W, b], lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY,
    )

    best_loss = torch.full((n_heads,), float("inf"), device=device)
    best_W = W.detach().cpu().clone()
    best_b = b.detach().cpu().clone()
    patience_left = torch.full((n_heads,), PROBE_PATIENCE, dtype=torch.int64, device=device)
    rng = np.random.default_rng(random_state)

    for _ in range(PROBE_MAX_EPOCHS):
        order = rng.permutation(len(fit_idx))
        for start in range(0, len(order), PROBE_BATCH_SIZE):
            idx = order[start:start + PROBE_BATCH_SIZE]
            xb = X_fit_t[:, idx, :]
            yb = y_fit_t[idx]
            preds = torch.einsum("hbd,hd->hb", xb, W) + b[:, None]
            loss = F.mse_loss(preds, yb.unsqueeze(0).expand(n_heads, -1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        monitor = _evaluate_multihead_regression_loss(W, b, X_val_t, y_val_t)
        if monitor is None:
            monitor = _evaluate_multihead_regression_loss(W, b, X_fit_t, y_fit_t)

        improved = monitor < (best_loss - 1e-5)
        if improved.any():
            W_cpu = W.detach().cpu()
            b_cpu = b.detach().cpu()
            improved_cpu = improved.detach().cpu()
            best_W[improved_cpu] = W_cpu[improved_cpu]
            best_b[improved_cpu] = b_cpu[improved_cpu]
            best_loss = torch.where(improved, monitor, best_loss)
            patience_left = torch.where(
                improved,
                torch.full_like(patience_left, PROBE_PATIENCE),
                patience_left - 1,
            )
        else:
            patience_left -= 1

        if bool((patience_left <= 0).all()):
            break

    W = best_W.to(device)
    b = best_b.to(device)
    return _predict_multihead_regressor(W, b, X_test_std, device)


def run_token_level_probes_batched(
    dataset,
    head_keys,
    *,
    sentence_split,
    device,
    random_state: int = 42,
    test_size: float = 0.2,
    head_chunk_size: int = TOKEN_HEAD_CHUNK_SIZE,
):
    """Train token-level probes for many heads together on the GPU."""
    if not head_keys:
        return {}

    ref = dataset[head_keys[0]]
    sentence_ids = ref.get("sentence_id")
    if sentence_split is None or sentence_ids is None or len(sentence_ids) == 0:
        return None

    train_mask = np.isin(sentence_ids, sentence_split["train"])
    test_mask = np.isin(sentence_ids, sentence_split["test"])
    if not train_mask.any() or not test_mask.any():
        return None

    per_head = {
        key: {"pos": None, "dep_rel": None, "depth": None}
        for key in head_keys
    }

    def _iter_head_chunks(desc):
        for start in tqdm(range(0, len(head_keys), head_chunk_size), desc=desc, unit="chunk"):
            yield head_keys[start:start + head_chunk_size]

    def _collect_chunk(chunk_keys):
        valid_keys = []
        arrays = []
        expected_len = len(sentence_ids)
        for key in chunk_keys:
            X = dataset[key]["X"]
            if X.shape[0] != expected_len or X.shape[0] < 20:
                continue
            valid_keys.append(key)
            arrays.append(X)
        if not valid_keys:
            return None, None
        return valid_keys, np.stack(arrays, axis=0)

    # POS
    le_pos = LabelEncoder()
    y_pos = le_pos.fit_transform(ref["pos"])
    y_pos_train = y_pos[train_mask]
    y_pos_test = y_pos[test_mask]
    if len(np.unique(y_pos_train)) >= 2 and len(y_pos_test) > 0:
        n_classes = len(np.unique(y_pos))
        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_pos_train).argmax()
        majority_acc = accuracy_score(y_pos_test, np.full_like(y_pos_test, majority_class))
        y_pos_shuffled = y_pos_train.copy()
        np.random.default_rng(random_state).shuffle(y_pos_shuffled)

        for chunk_keys in _iter_head_chunks("POS head chunks"):
            valid_keys, X_chunk = _collect_chunk(chunk_keys)
            if X_chunk is None:
                continue
            X_train = X_chunk[:, train_mask, :]
            X_test = X_chunk[:, test_mask, :]
            y_pred = _fit_multihead_linear_classifier_torch(
                X_train, y_pos_train, X_test,
                n_classes=n_classes,
                device=device,
                random_state=random_state,
            )
            y_rand = _fit_multihead_linear_classifier_torch(
                X_train, y_pos_shuffled, X_test,
                n_classes=n_classes,
                device=device,
                random_state=random_state,
            )
            for idx, key in enumerate(valid_keys):
                per_head[key]["pos"] = {
                    "accuracy": float(accuracy_score(y_pos_test, y_pred[idx])),
                    "f1_weighted": float(f1_score(y_pos_test, y_pred[idx], average="weighted", zero_division=0)),
                    "n_classes": int(n_classes),
                    "n_samples": int(X_chunk.shape[1]),
                    "random_baseline": float(random_acc),
                    "majority_baseline": float(majority_acc),
                    "random_label_baseline": float(accuracy_score(y_pos_test, y_rand[idx])),
                }

    # Dependency relation
    le_dep = LabelEncoder()
    y_dep = le_dep.fit_transform(ref["dep_rel"])
    y_dep_train = y_dep[train_mask]
    y_dep_test = y_dep[test_mask]
    if len(np.unique(y_dep_train)) >= 2 and len(y_dep_test) > 0:
        n_classes = len(np.unique(y_dep))
        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_dep_train).argmax()
        majority_acc = accuracy_score(y_dep_test, np.full_like(y_dep_test, majority_class))
        y_dep_shuffled = y_dep_train.copy()
        np.random.default_rng(random_state).shuffle(y_dep_shuffled)

        for chunk_keys in _iter_head_chunks("Dep head chunks"):
            valid_keys, X_chunk = _collect_chunk(chunk_keys)
            if X_chunk is None:
                continue
            X_train = X_chunk[:, train_mask, :]
            X_test = X_chunk[:, test_mask, :]
            y_pred = _fit_multihead_linear_classifier_torch(
                X_train, y_dep_train, X_test,
                n_classes=n_classes,
                device=device,
                random_state=random_state,
            )
            y_rand = _fit_multihead_linear_classifier_torch(
                X_train, y_dep_shuffled, X_test,
                n_classes=n_classes,
                device=device,
                random_state=random_state,
            )
            for idx, key in enumerate(valid_keys):
                per_head[key]["dep_rel"] = {
                    "accuracy": float(accuracy_score(y_dep_test, y_pred[idx])),
                    "f1_weighted": float(f1_score(y_dep_test, y_pred[idx], average="weighted", zero_division=0)),
                    "n_classes": int(n_classes),
                    "n_samples": int(X_chunk.shape[1]),
                    "random_baseline": float(random_acc),
                    "majority_baseline": float(majority_acc),
                    "random_label_baseline": float(accuracy_score(y_dep_test, y_rand[idx])),
                }

    # Depth
    y_depth = np.asarray(ref["depth"], dtype=np.float32)
    y_depth_train = y_depth[train_mask]
    y_depth_test = y_depth[test_mask]
    y_depth_shuffled = y_depth_train.copy()
    np.random.default_rng(random_state).shuffle(y_depth_shuffled)

    for chunk_keys in _iter_head_chunks("Depth head chunks"):
        valid_keys, X_chunk = _collect_chunk(chunk_keys)
        if X_chunk is None:
            continue
        X_train = X_chunk[:, train_mask, :]
        X_test = X_chunk[:, test_mask, :]
        y_pred = _fit_multihead_linear_regressor_torch(
            X_train, y_depth_train, X_test,
            device=device,
            random_state=random_state,
        )
        y_rand = _fit_multihead_linear_regressor_torch(
            X_train, y_depth_shuffled, X_test,
            device=device,
            random_state=random_state,
        )
        mean_pred = np.full_like(y_depth_test, np.mean(y_depth_train))
        baseline_mse = mean_squared_error(y_depth_test, mean_pred)

        for idx, key in enumerate(valid_keys):
            pred = y_pred[idx]
            if np.std(y_depth_test) > 0 and np.std(pred) > 0:
                spearman_r, spearman_p = stats.spearmanr(y_depth_test, pred)
            else:
                spearman_r, spearman_p = 0.0, 1.0
            per_head[key]["depth"] = {
                "mse": float(mean_squared_error(y_depth_test, pred)),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "n_samples": int(X_chunk.shape[1]),
                "mean_baseline_mse": float(baseline_mse),
                "random_label_mse": float(mean_squared_error(y_depth_test, y_rand[idx])),
            }

    return per_head


# ---------------------------------------------------------------------------
# Probing classifiers
# ---------------------------------------------------------------------------

class ProbingExperiment:
    """Train and evaluate linear probes for POS, dependency, and depth."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                 sentence_split=None, device=None):
        self.test_size = test_size
        self.random_state = random_state
        # If provided, train/test is done at the SENTENCE level so tokens from
        # the same sentence never leak across the split. This matches
        # probes.tex §3.3: "the same train/test split as the main translation
        # experiments".
        self.sentence_split = sentence_split
        self.device = device or _default_probe_device()

    # -- NaN sanitiser ------------------------------------------------------

    @staticmethod
    def _sanitise(X: np.ndarray, y: np.ndarray, sentence_ids=None):
        """Drop rows where any feature is NaN or Inf.

        Returns (X_clean, y_clean, sentence_ids_clean) or (None, None, None).
        """
        finite_mask = np.isfinite(X).all(axis=1)
        n_bad = (~finite_mask).sum()
        if n_bad:
            print(f"    [sanitise] dropping {n_bad} rows with NaN/Inf features")
        X_c, y_c = X[finite_mask], y[finite_mask]
        sid_c = sentence_ids[finite_mask] if sentence_ids is not None else None
        if len(X_c) < 20:
            return None, None, None
        return X_c, y_c, sid_c

    # -- Safe split helper --------------------------------------------------

    def _safe_train_test_split(self, X: np.ndarray, y: np.ndarray,
                                n_classes: int, sentence_ids=None):
        """Sentence-level split when self.sentence_split is set; otherwise
        stratified row-level split that drops singleton classes.

        Returns (X_train, X_test, y_train, y_test) or None if too few samples
        remain after filtering rare classes.
        """
        if self.sentence_split is not None and sentence_ids is not None:
            train_mask = np.isin(sentence_ids, self.sentence_split["train"])
            test_mask = np.isin(sentence_ids, self.sentence_split["test"])
            if not train_mask.any() or not test_mask.any():
                return None
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            if len(np.unique(y_train)) < 2 or len(y_test) == 0:
                return None
            return X_train, X_test, y_train, y_test

        class_counts = np.bincount(y)
        rare_classes = np.where(class_counts < 2)[0]

        if len(rare_classes) == 0:
            # All classes have at least 2 members — safe to stratify
            return train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if n_classes > 1 else None,
            )

        # Drop tokens whose class has only 1 member; they cannot be stratified
        mask = np.isin(y, rare_classes, invert=True)
        X_filt, y_filt = X[mask], y[mask]

        if len(X_filt) < 20:
            return None  # Caller treats this as an unusable probe

        n_classes_filt = len(np.unique(y_filt))
        return train_test_split(
            X_filt, y_filt,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_filt if n_classes_filt > 1 else None,
        )

    # -- POS tagging --------------------------------------------------------

    def probe_pos(self, X: np.ndarray, labels: list[str], sentence_ids=None):
        """Linear probe for POS classification.

        Returns dict with accuracy, f1, and baseline scores.
        """
        if len(X) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X, y, sentence_ids = self._sanitise(X, y, sentence_ids)
        if X is None:
            return None

        n_classes = len(np.unique(y))

        split = self._safe_train_test_split(X, y, n_classes, sentence_ids)
        if split is None:
            return None
        X_train, X_test, y_train, y_test = split

        y_pred = _fit_linear_classifier_torch(
            X_train,
            y_train,
            X_test,
            n_classes=n_classes,
            device=self.device,
            random_state=self.random_state,
        )

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Baselines
        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        # Random label baseline
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        y_rand = _fit_linear_classifier_torch(
            X_train,
            y_shuffled,
            X_test,
            n_classes=n_classes,
            device=self.device,
            random_state=self.random_state,
        )
        rand_label_acc = accuracy_score(y_test, y_rand)

        return {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "n_classes": n_classes,
            "n_samples": len(X),
            "random_baseline": float(random_acc),
            "majority_baseline": float(majority_acc),
            "random_label_baseline": float(rand_label_acc),
        }

    # -- Dependency relations -----------------------------------------------

    def probe_dependency(self, X: np.ndarray, labels: list[str],
                         sentence_ids=None):
        """Linear probe for dependency relation classification."""
        if len(X) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X, y, sentence_ids = self._sanitise(X, y, sentence_ids)
        if X is None:
            return None

        n_classes = len(np.unique(y))

        split = self._safe_train_test_split(X, y, n_classes, sentence_ids)
        if split is None:
            return None
        X_train, X_test, y_train, y_test = split

        y_pred = _fit_linear_classifier_torch(
            X_train,
            y_train,
            X_test,
            n_classes=n_classes,
            device=self.device,
            random_state=self.random_state,
        )

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        y_rand = _fit_linear_classifier_torch(
            X_train,
            y_shuffled,
            X_test,
            n_classes=n_classes,
            device=self.device,
            random_state=self.random_state,
        )
        rand_label_acc = accuracy_score(y_test, y_rand)

        return {
            "accuracy": float(acc),
            "f1_weighted": float(f1),
            "n_classes": n_classes,
            "n_samples": len(X),
            "random_baseline": float(random_acc),
            "majority_baseline": float(majority_acc),
            "random_label_baseline": float(rand_label_acc),
        }

    # -- Syntactic depth (regression) ---------------------------------------

    def probe_depth(self, X: np.ndarray, depths: np.ndarray,
                    sentence_ids=None):
        """Linear probe for syntactic depth (regression)."""
        if len(X) < 20:
            return None

        X, depths, sentence_ids = self._sanitise(X, depths, sentence_ids)
        if X is None:
            return None

        if self.sentence_split is not None and sentence_ids is not None:
            train_mask = np.isin(sentence_ids, self.sentence_split["train"])
            test_mask = np.isin(sentence_ids, self.sentence_split["test"])
            if not train_mask.any() or not test_mask.any():
                return None
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = depths[train_mask], depths[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, depths, test_size=self.test_size,
                random_state=self.random_state,
            )

        y_pred = _fit_linear_regressor_torch(
            X_train,
            y_train,
            X_test,
            device=self.device,
            random_state=self.random_state,
        )

        mse = mean_squared_error(y_test, y_pred)
        # Spearman correlation for ordinal depth
        if np.std(y_test) > 0 and np.std(y_pred) > 0:
            spearman_r, spearman_p = stats.spearmanr(y_test, y_pred)
        else:
            spearman_r, spearman_p = 0.0, 1.0

        # Baseline: always predict mean depth
        mean_pred = np.full_like(y_test, np.mean(y_train))
        baseline_mse = mean_squared_error(y_test, mean_pred)

        # Random label baseline
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        y_rand = _fit_linear_regressor_torch(
            X_train,
            y_shuffled,
            X_test,
            device=self.device,
            random_state=self.random_state,
        )
        rand_mse = mean_squared_error(y_test, y_rand)

        return {
            "mse": float(mse),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "n_samples": len(X),
            "mean_baseline_mse": float(baseline_mse),
            "random_label_mse": float(rand_mse),
        }


# ---------------------------------------------------------------------------
# Pairwise dependency structure probing
# ---------------------------------------------------------------------------

class PairwiseDependencyProber:
    """Pairwise probing for dependency arc prediction and relation classification.

    Tests whether head representations encode relational syntactic structure
    by predicting dependency arcs between token pairs.
    """

    COMBINATION_METHODS = ("concat", "diff", "product", "full")

    def __init__(
        self,
        combination: str = "concat",
        test_size: float = 0.2,
        random_state: int = 42,
        max_pairs_per_sentence: int = None,
        sentence_split=None,
        device=None,
    ):
        assert combination in self.COMBINATION_METHODS
        self.combination = combination
        self.test_size = test_size
        self.random_state = random_state
        self.max_pairs_per_sentence = max_pairs_per_sentence
        self.sentence_split = sentence_split
        self.device = device or _default_probe_device()

    # -- Representation combination -----------------------------------------

    @staticmethod
    def combine_concat(h_i, h_j):
        return np.concatenate([h_i, h_j], axis=-1)

    @staticmethod
    def combine_diff(h_i, h_j):
        return h_i - h_j

    @staticmethod
    def combine_product(h_i, h_j):
        return h_i * h_j

    @staticmethod
    def combine_full(h_i, h_j):
        return np.concatenate([h_i, h_j, h_i - h_j, h_i * h_j], axis=-1)

    def combine(self, h_i, h_j):
        fn = {
            "concat": self.combine_concat,
            "diff": self.combine_diff,
            "product": self.combine_product,
            "full": self.combine_full,
        }[self.combination]
        return fn(h_i, h_j)

    # -- NaN sanitiser ------------------------------------------------------

    @staticmethod
    def _sanitise(X, y, sentence_ids=None):
        finite_mask = np.isfinite(X).all(axis=1)
        n_bad = (~finite_mask).sum()
        if n_bad:
            print(f"    [pairwise sanitise] dropping {n_bad} rows with NaN/Inf")
        X_c, y_c = X[finite_mask], y[finite_mask]
        sid_c = sentence_ids[finite_mask] if sentence_ids is not None else None
        if len(X_c) < 20:
            return None, None, None
        return X_c, y_c, sid_c

    # -- Binary arc prediction ----------------------------------------------

    def probe_arc(self, X_pairs, y_arc, sentence_ids=None):
        """Binary classifier: does token i depend on token j?

        Returns dict with accuracy, precision, recall, F1, and baselines.
        """
        if len(X_pairs) < 20:
            return None

        X_pairs, y_arc, sentence_ids = self._sanitise(X_pairs, y_arc, sentence_ids)
        if X_pairs is None:
            return None

        n_pos = int(y_arc.sum())
        n_neg = len(y_arc) - n_pos
        if n_pos < 2 or n_neg < 2:
            return None

        if self.sentence_split is not None and sentence_ids is not None:
            train_mask = np.isin(sentence_ids, self.sentence_split["train"])
            test_mask = np.isin(sentence_ids, self.sentence_split["test"])
            if not train_mask.any() or not test_mask.any():
                return None
            X_train, X_test = X_pairs[train_mask], X_pairs[test_mask]
            y_train, y_test = y_arc[train_mask], y_arc[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                return None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_pairs, y_arc,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_arc,
            )

        weights = _balanced_class_weights(y_train, 2)
        y_pred = _fit_linear_classifier_torch(
            X_train,
            y_train,
            X_test,
            n_classes=2,
            device=self.device,
            random_state=self.random_state,
            class_weights=weights,
        )

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Baselines
        n_test = len(y_test)
        n_test_pos = int(y_test.sum())
        random_acc = (n_test_pos / n_test) if n_test > 0 else 0.0
        majority_acc = max(n_test_pos, n_test - n_test_pos) / n_test

        # Random label baseline
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        y_rand = _fit_linear_classifier_torch(
            X_train,
            y_shuffled,
            X_test,
            n_classes=2,
            device=self.device,
            random_state=self.random_state,
        )
        rand_f1 = f1_score(y_test, y_rand, zero_division=0)

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "n_pairs": len(X_pairs),
            "n_positive": int(np.sum(y_arc)),
            "positive_rate": float(np.mean(y_arc)),
            "random_baseline_acc": float(random_acc),
            "majority_baseline_acc": float(majority_acc),
            "random_label_f1": float(rand_f1),
        }

    # -- Relation type classification (positive pairs only) -----------------

    def probe_relation(self, X_pairs, y_rel_labels, sentence_ids=None):
        """Multi-class classifier for dependency relation type.

        Trained only on positive pairs (where a dependency arc exists).
        Returns dict with accuracy, macro F1, and baselines.
        """
        if len(X_pairs) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(y_rel_labels)

        X_pairs, y, sentence_ids = self._sanitise(X_pairs, y, sentence_ids)
        if X_pairs is None:
            return None

        n_classes = len(np.unique(y))
        if n_classes < 2:
            return None

        kept_class_ids = np.unique(y)

        # Drop singleton classes for stratified split
        class_counts = np.bincount(y)
        rare = np.where(class_counts < 2)[0]
        if len(rare) > 0:
            mask = np.isin(y, rare, invert=True)
            X_pairs, y = X_pairs[mask], y[mask]
            if sentence_ids is not None:
                sentence_ids = sentence_ids[mask]
            if len(X_pairs) < 20:
                return None
            kept_class_ids = np.unique(y)
            remap = {old: new for new, old in enumerate(kept_class_ids.tolist())}
            y = np.array([remap[int(label)] for label in y], dtype=np.int64)
            n_classes = len(kept_class_ids)
        else:
            kept_class_ids = np.unique(y)

        if self.sentence_split is not None and sentence_ids is not None:
            train_mask = np.isin(sentence_ids, self.sentence_split["train"])
            test_mask = np.isin(sentence_ids, self.sentence_split["test"])
            if not train_mask.any() or not test_mask.any():
                return None
            X_train, X_test = X_pairs[train_mask], X_pairs[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            if len(np.unique(y_train)) < 2 or len(y_test) == 0:
                return None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_pairs, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

        y_pred = _fit_linear_classifier_torch(
            X_train,
            y_train,
            X_test,
            n_classes=n_classes,
            device=self.device,
            random_state=self.random_state,
        )

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        y_rand = _fit_linear_classifier_torch(
            X_train,
            y_shuffled,
            X_test,
            n_classes=n_classes,
            device=self.device,
            random_state=self.random_state,
        )
        rand_label_acc = accuracy_score(y_test, y_rand)

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "n_classes": n_classes,
            "n_samples": len(X_pairs),
            "classes": le.inverse_transform(kept_class_ids).tolist(),
            "random_baseline": float(random_acc),
            "majority_baseline": float(majority_acc),
            "random_label_baseline": float(rand_label_acc),
        }



# ---------------------------------------------------------------------------
# Pairwise dataset construction
# ---------------------------------------------------------------------------

def _pairwise_feature_dim(combination: str, d_head: int) -> int:
    if combination == "concat":
        return d_head * 2
    if combination in ("diff", "product"):
        return d_head
    return d_head * 4


def collect_pairwise_sentence_data(
    extractor: HeadRepresentationExtractor,
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    max_sentences: int = None,
    batch_size: int = 8,
    head_keys=None,
):
    """Cache aligned sentence-level data for pairwise probing.

    This keeps one copy of the per-sentence head representations in memory and
    lets us materialise one head's pairwise dataset at a time, which avoids the
    enormous RAM blow-up from storing pairwise matrices for all 144 heads at
    once.
    """
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]

    sentence_data = []
    skipped = 0
    head_key_set = set(head_keys) if head_keys is not None else None
    progress = tqdm(
        total=len(scrambled_sentences),
        desc="Pairwise sentence cache",
        unit="sent",
    )
    iterator = zip(scrambled_sentences, original_sentences)
    for i, (scrambled, original) in enumerate(iterator):
        labels = labeler.extract_labels(original)
        head_reps, token_ids, tokens, offsets = extractor.extract(scrambled)
        if head_reps is None:
            skipped += 1
            progress.update(1)
            continue

        if head_key_set is not None:
            head_reps = {
                key: np.asarray(head_reps[key], dtype=PAIRWISE_FEATURE_DTYPE)
                for key in head_key_set
            }

        aligned = align_scrambled_to_original_by_identity(
            scrambled, offsets, labeler, labels,
        )
        if len(aligned) < 2:
            skipped += 1
            progress.update(1)
            continue

        sentence_data.append({
            "aligned": aligned,
            "head_reps": head_reps,
            "n_aligned": len(aligned),
            "sentence_id": i,
        })
        progress.update(1)

    progress.close()

    if skipped:
        print(
            f"  Skipped {skipped}/{len(scrambled_sentences)} sentences (pairwise alignment)",
            flush=True,
        )

    return sentence_data


def build_pairwise_dataset_for_head(
    sentence_data,
    layer_idx: int,
    head_idx: int,
    *,
    combination: str = "concat",
    d_head: int,
):
    """Materialise one head's pairwise dataset from cached sentence data."""
    combiner = PairwiseDependencyProber(combination=combination)
    d_comb = _pairwise_feature_dim(combination, d_head)
    total_pairs = 0
    total_pos_pairs = 0
    for sent in sentence_data:
        aligned = sent["aligned"]
        n_tok = sent["n_aligned"]
        total_pairs += n_tok * (n_tok - 1)
        aligned_word_indices = {entry["word_idx"] for entry in aligned}
        for entry in aligned:
            if entry["head_idx"] in aligned_word_indices and entry["head_idx"] != entry["word_idx"]:
                total_pos_pairs += 1

    if total_pairs == 0:
        return {
            "X_pairs": np.empty((0, d_comb), dtype=PAIRWISE_FEATURE_DTYPE),
            "y_arc": np.empty(0, dtype=np.int32),
            "X_pairs_pos": np.empty((0, d_comb), dtype=PAIRWISE_FEATURE_DTYPE),
            "y_rel_pos": [],
            "distances": np.empty(0, dtype=np.int32),
            "sentence_id": np.empty(0, dtype=np.int32),
            "sentence_id_pos": np.empty(0, dtype=np.int32),
        }

    X_pairs = np.empty((total_pairs, d_comb), dtype=PAIRWISE_FEATURE_DTYPE)
    y_arc = np.empty(total_pairs, dtype=np.int32)
    distances = np.empty(total_pairs, dtype=np.int32)
    sentence_ids = np.empty(total_pairs, dtype=np.int32)
    X_pairs_pos = np.empty((total_pos_pairs, d_comb), dtype=PAIRWISE_FEATURE_DTYPE)
    y_rel_pos = []
    sentence_ids_pos = np.empty(total_pos_pairs, dtype=np.int32)

    pair_cursor = 0
    pos_cursor = 0
    for sent in sentence_data:
        aligned = sent["aligned"]
        n_tok = sent["n_aligned"]
        reps = sent["head_reps"][(layer_idx, head_idx)]
        sid = sent["sentence_id"]

        for idx_i in range(n_tok):
            for idx_j in range(n_tok):
                if idx_i == idx_j:
                    continue

                sw_i = aligned[idx_i]["subword_idx"]
                sw_j = aligned[idx_j]["subword_idx"]
                if sw_i >= reps.shape[0] or sw_j >= reps.shape[0]:
                    continue

                h_i = reps[sw_i]
                h_j = reps[sw_j]
                c_ij = combiner.combine(h_i, h_j)

                head_of_i = aligned[idx_i]["head_idx"]
                word_idx_j = aligned[idx_j]["word_idx"]
                is_arc = int(head_of_i == word_idx_j)

                X_pairs[pair_cursor] = c_ij
                y_arc[pair_cursor] = is_arc
                distances[pair_cursor] = abs(
                    aligned[idx_i]["scrambled_word_idx"]
                    - aligned[idx_j]["scrambled_word_idx"]
                )
                sentence_ids[pair_cursor] = sid
                pair_cursor += 1

                if is_arc:
                    X_pairs_pos[pos_cursor] = c_ij
                    y_rel_pos.append(aligned[idx_i]["dep_rel"])
                    sentence_ids_pos[pos_cursor] = sid
                    pos_cursor += 1

    if pair_cursor != total_pairs:
        X_pairs = X_pairs[:pair_cursor]
        y_arc = y_arc[:pair_cursor]
        distances = distances[:pair_cursor]
        sentence_ids = sentence_ids[:pair_cursor]
    if pos_cursor != total_pos_pairs:
        X_pairs_pos = X_pairs_pos[:pos_cursor]
        sentence_ids_pos = sentence_ids_pos[:pos_cursor]
        y_rel_pos = y_rel_pos[:pos_cursor]

    return {
        "X_pairs": X_pairs,
        "y_arc": y_arc,
        "X_pairs_pos": X_pairs_pos,
        "y_rel_pos": y_rel_pos,
        "distances": distances,
        "sentence_id": sentence_ids,
        "sentence_id_pos": sentence_ids_pos,
    }


def _combine_pair_features(h_i: np.ndarray, h_j: np.ndarray, combination: str):
    if combination == "concat":
        return np.concatenate([h_i, h_j], axis=-1)
    if combination == "diff":
        return h_i - h_j
    if combination == "product":
        return h_i * h_j
    return np.concatenate([h_i, h_j, h_i - h_j, h_i * h_j], axis=-1)


def _sentence_pair_arrays(sent, layer_idx: int, head_idx: int, combination: str):
    aligned = sent["aligned"]
    n_tok = sent["n_aligned"]
    d_comb = _pairwise_feature_dim(combination, sent["head_reps"][(layer_idx, head_idx)].shape[1])
    if n_tok < 2:
        return (
            np.empty((0, d_comb), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            [],
        )

    reps = np.asarray(sent["head_reps"][(layer_idx, head_idx)], dtype=np.float32)
    sw_idx = np.array([entry["subword_idx"] for entry in aligned], dtype=np.int32)
    valid = sw_idx < reps.shape[0]
    if valid.sum() < 2:
        return (
            np.empty((0, d_comb), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            [],
        )

    reps_sub = reps[sw_idx[valid]]
    aligned_valid = [entry for entry, keep in zip(aligned, valid) if keep]
    n_tok = len(aligned_valid)
    if n_tok < 2:
        return (
            np.empty((0, d_comb), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            [],
        )

    rows, cols = np.where(~np.eye(n_tok, dtype=bool))
    h_i = reps_sub[rows]
    h_j = reps_sub[cols]
    X_pairs = _combine_pair_features(h_i, h_j, combination).astype(np.float32, copy=False)

    word_idx = np.array([entry["word_idx"] for entry in aligned_valid], dtype=np.int32)
    head_idx_arr = np.array([entry["head_idx"] for entry in aligned_valid], dtype=np.int32)
    y_arc = (head_idx_arr[rows] == word_idx[cols]).astype(np.int32)
    dep_rels = [entry["dep_rel"] for entry in aligned_valid]
    y_rel = [dep_rels[row] for row, is_arc in zip(rows.tolist(), y_arc.tolist()) if is_arc]
    return X_pairs, y_arc, y_rel


def build_positive_pairwise_dataset_for_head(
    sentence_data,
    layer_idx: int,
    head_idx: int,
    *,
    combination: str = "concat",
):
    d_head = next(iter(sentence_data))["head_reps"][(layer_idx, head_idx)].shape[1] if sentence_data else 64
    d_comb = _pairwise_feature_dim(combination, d_head)
    X_pos_list = []
    y_rel_pos = []
    sentence_ids_pos = []

    for sent in sentence_data:
        X_pairs, y_arc, y_rel = _sentence_pair_arrays(sent, layer_idx, head_idx, combination)
        if len(y_arc) == 0:
            continue
        pos_mask = y_arc.astype(bool)
        if pos_mask.any():
            X_pos = X_pairs[pos_mask].astype(PAIRWISE_FEATURE_DTYPE, copy=False)
            X_pos_list.append(X_pos)
            y_rel_pos.extend(y_rel)
            sentence_ids_pos.extend([sent["sentence_id"]] * len(y_rel))

    if not X_pos_list:
        return {
            "X_pairs_pos": np.empty((0, d_comb), dtype=PAIRWISE_FEATURE_DTYPE),
            "y_rel_pos": [],
            "sentence_id_pos": np.empty(0, dtype=np.int32),
        }

    return {
        "X_pairs_pos": np.concatenate(X_pos_list, axis=0),
        "y_rel_pos": y_rel_pos,
        "sentence_id_pos": np.array(sentence_ids_pos, dtype=np.int32),
    }


def probe_arc_streaming_for_head(
    sentence_data,
    layer_idx: int,
    head_idx: int,
    *,
    combination: str,
    sentence_split,
    device,
    random_state: int = 42,
):
    available_train = [
        sent for sent in sentence_data
        if sent["sentence_id"] in set(sentence_split["train"])
    ]
    available_test = [
        sent for sent in sentence_data
        if sent["sentence_id"] in set(sentence_split["test"])
    ]
    if not available_train or not available_test:
        return None

    train_sentence_ids = np.array([sent["sentence_id"] for sent in available_train], dtype=np.int32)
    fit_ids = train_sentence_ids
    val_ids = None
    if len(train_sentence_ids) >= 10:
        fit_ids, val_ids = train_test_split(
            train_sentence_ids,
            test_size=0.1,
            random_state=random_state,
            shuffle=True,
        )
    fit_id_set = set(np.asarray(fit_ids).tolist())
    val_id_set = set(np.asarray(val_ids).tolist()) if val_ids is not None else None

    fit_sents = [sent for sent in available_train if sent["sentence_id"] in fit_id_set]
    val_sents = (
        [sent for sent in available_train if sent["sentence_id"] in val_id_set]
        if val_id_set is not None else None
    )

    def _iter_arc_batches(sentences, *, shuffle=False, shuffle_labels=False):
        rng = np.random.default_rng(random_state)
        order = np.arange(len(sentences))
        if shuffle:
            rng.shuffle(order)
        for idx in order:
            X_pairs, y_arc, _ = _sentence_pair_arrays(
                sentences[idx], layer_idx, head_idx, combination,
            )
            if len(y_arc) == 0:
                continue
            if shuffle_labels:
                y_arc = y_arc.copy()
                rng.shuffle(y_arc)
            for start in range(0, len(y_arc), PROBE_BATCH_SIZE):
                stop = min(start + PROBE_BATCH_SIZE, len(y_arc))
                yield X_pairs[start:stop], y_arc[start:stop]

    n_train = 0
    n_positive = 0
    for _, y_batch in _iter_arc_batches(fit_sents):
        n_train += len(y_batch)
        n_positive += int(y_batch.sum())
    n_negative = n_train - n_positive
    if n_train < 20 or n_positive < 2 or n_negative < 2:
        return None

    d_comb = _pairwise_feature_dim(
        combination,
        next(iter(sentence_data))["head_reps"][(layer_idx, head_idx)].shape[1],
    )
    weight_vec = _balanced_class_weights(
        np.concatenate([
            np.ones(n_positive, dtype=np.int32),
            np.zeros(n_negative, dtype=np.int32),
        ]),
        2,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weight_vec, dtype=torch.float32, device=device),
    )
    torch.manual_seed(random_state)
    model = nn.Linear(d_comb, 2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY,
    )

    best_state = None
    best_loss = float("inf")
    patience_left = PROBE_PATIENCE

    def _eval_loss(sentences, *, shuffle_labels=False):
        losses = []
        total = 0
        with torch.no_grad():
            for xb_np, yb_np in _iter_arc_batches(sentences, shuffle=False, shuffle_labels=shuffle_labels):
                xb = torch.from_numpy(xb_np.astype(np.float32, copy=False)).to(device)
                yb = torch.from_numpy(yb_np.astype(np.int64, copy=False)).to(device)
                logits = model(xb)
                losses.append(float(criterion(logits, yb).detach().cpu()) * len(yb_np))
                total += len(yb_np)
        return (sum(losses) / total) if total else None

    for _ in range(PROBE_MAX_EPOCHS):
        model.train()
        for xb_np, yb_np in _iter_arc_batches(fit_sents, shuffle=True):
            xb = torch.from_numpy(xb_np.astype(np.float32, copy=False)).to(device)
            yb = torch.from_numpy(yb_np.astype(np.int64, copy=False)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        monitor = _eval_loss(val_sents if val_sents else fit_sents)
        if monitor is None:
            break
        if monitor < best_loss - 1e-4:
            best_loss = monitor
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience_left = PROBE_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for xb_np, yb_np in _iter_arc_batches(available_test):
            xb = torch.from_numpy(xb_np.astype(np.float32, copy=False)).to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.append(yb_np)
            y_pred.append(pred)
    if not y_true:
        return None
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Random-label baseline via shuffled training labels in the same streaming setup.
    rand_model = nn.Linear(d_comb, 2).to(device)
    rand_optimizer = torch.optim.AdamW(
        rand_model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY,
    )
    best_state = None
    best_loss = float("inf")
    patience_left = PROBE_PATIENCE

    def _eval_rand_loss(sentences):
        losses = []
        total = 0
        with torch.no_grad():
            for xb_np, yb_np in _iter_arc_batches(sentences, shuffle=False, shuffle_labels=True):
                xb = torch.from_numpy(xb_np.astype(np.float32, copy=False)).to(device)
                yb = torch.from_numpy(yb_np.astype(np.int64, copy=False)).to(device)
                logits = rand_model(xb)
                losses.append(float(criterion(logits, yb).detach().cpu()) * len(yb_np))
                total += len(yb_np)
        return (sum(losses) / total) if total else None

    for _ in range(PROBE_MAX_EPOCHS):
        rand_model.train()
        for xb_np, yb_np in _iter_arc_batches(fit_sents, shuffle=True, shuffle_labels=True):
            xb = torch.from_numpy(xb_np.astype(np.float32, copy=False)).to(device)
            yb = torch.from_numpy(yb_np.astype(np.int64, copy=False)).to(device)
            rand_optimizer.zero_grad(set_to_none=True)
            loss = criterion(rand_model(xb), yb)
            loss.backward()
            rand_optimizer.step()

        monitor = _eval_rand_loss(val_sents if val_sents else fit_sents)
        if monitor is None:
            break
        if monitor < best_loss - 1e-4:
            best_loss = monitor
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in rand_model.state_dict().items()
            }
            patience_left = PROBE_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        rand_model.load_state_dict(best_state)

    y_rand = []
    with torch.no_grad():
        rand_model.eval()
        for xb_np, _ in _iter_arc_batches(available_test):
            xb = torch.from_numpy(xb_np.astype(np.float32, copy=False)).to(device)
            y_rand.append(rand_model(xb).argmax(dim=1).cpu().numpy())
    y_rand = np.concatenate(y_rand) if y_rand else np.empty(0, dtype=np.int64)

    n_test = len(y_true)
    n_test_pos = int(y_true.sum())
    random_acc = (n_test_pos / n_test) if n_test > 0 else 0.0
    majority_acc = max(n_test_pos, n_test - n_test_pos) / n_test if n_test > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_pairs": int(n_test),
        "n_positive": int(n_test_pos),
        "positive_rate": float(np.mean(y_true)) if n_test > 0 else 0.0,
        "random_baseline_acc": float(random_acc),
        "majority_baseline_acc": float(majority_acc),
        "random_label_f1": float(f1_score(y_true, y_rand, zero_division=0)) if len(y_rand) else 0.0,
    }


# ---------------------------------------------------------------------------
# Pairwise baselines
# ---------------------------------------------------------------------------

def compute_pairwise_baselines(
    extractor: HeadRepresentationExtractor,
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    combination: str = "concat",
    max_sentences: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    sentence_split=None,
):
    """Word-embedding pairwise baseline and distance baseline."""
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]

    combiner = PairwiseDependencyProber(combination=combination)
    wte = extractor.model.transformer.wte

    X_emb_pairs = []
    y_arc_all = []
    y_rel_pos = []
    X_emb_pos = []
    distances_all = []
    sentence_ids_all = []
    sentence_ids_pos = []

    iterator = tqdm(
        zip(scrambled_sentences, original_sentences),
        total=len(scrambled_sentences),
        desc="Pairwise baseline",
        unit="sent",
    )
    for i, (scrambled, original) in enumerate(iterator):

        labels = labeler.extract_labels(original)
        encoded = extractor.tokenizer(
            scrambled,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = [tuple(s) for s in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {k: v.to(extractor.device) for k, v in encoded.items()}

        if inputs["input_ids"].shape[1] == 0:
            continue

        with torch.no_grad():
            embeddings = wte(inputs["input_ids"])[0].cpu().numpy()

        aligned = align_scrambled_to_original_by_identity(
            scrambled, offsets, labeler, labels,
        )
        if len(aligned) < 2:
            continue

        for idx_i in range(len(aligned)):
            for idx_j in range(len(aligned)):
                if idx_i == idx_j:
                    continue

                sw_i = aligned[idx_i]["subword_idx"]
                sw_j = aligned[idx_j]["subword_idx"]
                if sw_i >= embeddings.shape[0] or sw_j >= embeddings.shape[0]:
                    continue

                c_ij = combiner.combine(embeddings[sw_i], embeddings[sw_j])

                head_of_i = aligned[idx_i]["head_idx"]
                word_idx_j = aligned[idx_j]["word_idx"]
                is_arc = int(head_of_i == word_idx_j)
                rel_label = aligned[idx_i]["dep_rel"] if is_arc else "NO_ARC"

                X_emb_pairs.append(c_ij)
                y_arc_all.append(is_arc)
                distances_all.append(abs(
                    aligned[idx_i]["scrambled_word_idx"]
                    - aligned[idx_j]["scrambled_word_idx"]
                ))
                sentence_ids_all.append(i)

                if is_arc:
                    X_emb_pos.append(c_ij)
                    y_rel_pos.append(rel_label)
                    sentence_ids_pos.append(i)

    result = {"word_emb_arc": None, "word_emb_rel": None, "distance_arc": None}

    if len(X_emb_pairs) < 20:
        return result

    X_emb_pairs = np.stack(X_emb_pairs)
    y_arc_all = np.array(y_arc_all, dtype=np.int32)
    distances_all = np.array(distances_all, dtype=np.int32)
    sentence_ids_all = np.array(sentence_ids_all, dtype=np.int32)
    sentence_ids_pos = np.array(sentence_ids_pos, dtype=np.int32)

    prober = PairwiseDependencyProber(
        combination=combination,
        test_size=test_size,
        random_state=random_state,
        sentence_split=sentence_split,
        device=extractor.device,
    )

    # Word-embedding arc baseline
    result["word_emb_arc"] = prober.probe_arc(
        X_emb_pairs, y_arc_all, sentence_ids_all,
    )

    # Word-embedding relation baseline (positive pairs only)
    if len(X_emb_pos) >= 20:
        X_emb_pos = np.stack(X_emb_pos)
        result["word_emb_rel"] = prober.probe_relation(
            X_emb_pos, y_rel_pos, sentence_ids_pos,
        )

    # Distance baseline: predict arc from |i-j| alone
    dist_features = distances_all.reshape(-1, 1).astype(np.float32)
    result["distance_arc"] = prober.probe_arc(
        dist_features, y_arc_all, sentence_ids_all,
    )

    return result


# ---------------------------------------------------------------------------
# Word-form baseline (probe on input embeddings instead of head outputs)
# ---------------------------------------------------------------------------

def compute_word_embedding_baseline(
    extractor: HeadRepresentationExtractor,
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    max_sentences: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    sentence_split=None,
):
    """Train probes on word embeddings (wte) as a control baseline."""
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]

    X_all, pos_all, dep_all, depth_all = [], [], [], []
    sent_ids_all = []
    wte = extractor.model.transformer.wte  # word token embedding layer

    iterator = tqdm(
        zip(scrambled_sentences, original_sentences),
        total=len(scrambled_sentences),
        desc="Word-emb baseline",
        unit="sent",
    )
    for i, (scrambled, original) in enumerate(iterator):

        labels = labeler.extract_labels(original)
        encoded = extractor.tokenizer(
            scrambled,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = [tuple(s) for s in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {k: v.to(extractor.device) for k, v in encoded.items()}

        if inputs["input_ids"].shape[1] == 0:
            continue

        with torch.no_grad():
            embeddings = wte(inputs["input_ids"])[0].cpu().numpy()  # (seq, emb)

        aligned = align_scrambled_to_original_by_identity(
            scrambled, offsets, labeler, labels,
        )

        for entry in aligned:
            sw_idx = entry["subword_idx"]
            if sw_idx < embeddings.shape[0]:
                X_all.append(embeddings[sw_idx])
                pos_all.append(entry["pos"])
                dep_all.append(entry["dep_rel"])
                depth_all.append(entry["depth"])
                sent_ids_all.append(i)

    if len(X_all) < 20:
        return {"pos": None, "dep_rel": None, "depth": None}

    X_all = np.stack(X_all)
    depth_all = np.array(depth_all, dtype=np.float32)
    sent_ids_all = np.array(sent_ids_all, dtype=np.int32)

    prober = ProbingExperiment(
        test_size=test_size,
        random_state=random_state,
        sentence_split=sentence_split,
        device=extractor.device,
    )
    return {
        "pos": prober.probe_pos(X_all, pos_all, sent_ids_all),
        "dep_rel": prober.probe_dependency(X_all, dep_all, sent_ids_all),
        "depth": prober.probe_depth(X_all, depth_all, sent_ids_all),
    }


# ---------------------------------------------------------------------------
# Full probing pipeline
# ---------------------------------------------------------------------------

def run_probing_pipeline(
    model_path: str,
    tokenizer_path: str,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    labeler: SyntacticLabeler,
    model_label: str = "model",
    max_sentences: int = None,
    device=None,
    sentence_split=None,
    pairwise_max_sentences: int = None,
    skip_pairwise: bool = False,
    batch_size: int = 8,
    pairwise_head_chunk_size: int = PAIRWISE_HEAD_CHUNK_SIZE,
):

    print(f"\n{'='*60}")
    print(f"  Probing: {model_label} ({model_path})")
    print(f"{'='*60}")

    extractor = HeadRepresentationExtractor(
        model_path, tokenizer_path=tokenizer_path, device=device,
    )

    # Build dataset
    print("\n  Building probing dataset...")
    dataset = build_probing_dataset(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
    )

    # Word-embedding baseline (token-level)
    print("\n  Computing word-embedding baseline...")
    emb_baseline = compute_word_embedding_baseline(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
        sentence_split=sentence_split,
    )

    # Probe each head (token-level)
    n_layers = extractor.n_layers
    n_heads = extractor.n_heads
    total = n_layers * n_heads
    head_keys = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    per_head = run_token_level_probes_batched(
        dataset,
        head_keys,
        sentence_split=sentence_split,
        device=extractor.device,
    )

    if per_head is None:
        print("\n  Falling back to per-head token probing...", flush=True)
        prober = ProbingExperiment(
            sentence_split=sentence_split,
            device=extractor.device,
        )
        per_head = {}
        for l, h in tqdm(head_keys, total=total, desc="Token-level heads", unit="head"):
            data = dataset[(l, h)]
            X = data["X"]
            if X.shape[0] < 20:
                per_head[(l, h)] = {"pos": None, "dep_rel": None, "depth": None}
                continue

            sid = data.get("sentence_id")
            pos_result = prober.probe_pos(X, data["pos"], sid)
            dep_result = prober.probe_dependency(X, data["dep_rel"], sid)
            depth_result = prober.probe_depth(X, data["depth"], sid)

            per_head[(l, h)] = {
                "pos": pos_result,
                "dep_rel": dep_result,
                "depth": depth_result,
            }

    # Free large token-level data before the pairwise stage.
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_head_pairwise = {}
    pw_baselines = {}
    pairwise_budget = pairwise_max_sentences
    if pairwise_budget is None:
        pairwise_budget = max_sentences

    if skip_pairwise:
        print("\n  Skipping pairwise probing (--skip_pairwise)", flush=True)
    else:
        if pairwise_budget is not None and max_sentences is not None and pairwise_budget < max_sentences:
            print(
                f"\n  Pairwise stages limited to {pairwise_budget}/{max_sentences} sentences",
                flush=True,
            )

        print("\n  Computing pairwise baselines...", flush=True)
        pw_baselines = compute_pairwise_baselines(
            extractor, labeler, scrambled_sentences, original_sentences,
            combination="concat",
            max_sentences=pairwise_budget,
            sentence_split=sentence_split,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pw_prober = PairwiseDependencyProber(
            combination="concat",
            sentence_split=sentence_split,
            device=extractor.device,
        )

        for chunk_start in tqdm(
            range(0, total, pairwise_head_chunk_size),
            total=(total + pairwise_head_chunk_size - 1) // pairwise_head_chunk_size,
            desc="Pairwise head chunks",
            unit="chunk",
        ):
            chunk_keys = head_keys[chunk_start:chunk_start + pairwise_head_chunk_size]
            print(
                f"\n  Caching pairwise sentence data for heads {chunk_start + 1}-{chunk_start + len(chunk_keys)}",
                flush=True,
            )
            pairwise_sentence_data = collect_pairwise_sentence_data(
                extractor, labeler, scrambled_sentences, original_sentences,
                max_sentences=pairwise_budget,
                batch_size=batch_size,
                head_keys=chunk_keys,
            )

            for l, h in chunk_keys:
                arc_result = probe_arc_streaming_for_head(
                    pairwise_sentence_data,
                    l,
                    h,
                    combination="concat",
                    sentence_split=sentence_split,
                    device=extractor.device,
                )
                pw_pos = build_positive_pairwise_dataset_for_head(
                    pairwise_sentence_data,
                    l,
                    h,
                    combination="concat",
                )
                rel_result = pw_prober.probe_relation(
                    pw_pos["X_pairs_pos"], pw_pos["y_rel_pos"],
                    pw_pos.get("sentence_id_pos"),
                )

                per_head_pairwise[(l, h)] = {
                    "arc": arc_result,
                    "relation": rel_result,
                }

                del pw_pos

            del pairwise_sentence_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Layer-wise summaries
    layer_summary = _compute_layer_summary(per_head, n_layers, n_heads)
    layer_summary_pairwise = _compute_pairwise_layer_summary(
        per_head_pairwise, n_layers, n_heads
    )

    # Clean up
    extractor.remove_hooks()
    del extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "per_head": per_head,
        "per_head_pairwise": per_head_pairwise,
        "layer_summary": layer_summary,
        "layer_summary_pairwise": layer_summary_pairwise,
        "word_embedding_baseline": emb_baseline,
        "pairwise_baselines": pw_baselines,
    }


def _compute_layer_summary(per_head, n_layers, n_heads):
    """Aggregate probing results by layer (Eq. in paper)."""
    summary = {}
    for prop in ["pos", "dep_rel", "depth"]:
        layer_means = []
        for l in range(n_layers):
            vals = []
            for h in range(n_heads):
                result = per_head.get((l, h), {}).get(prop)
                if result is None:
                    continue
                if prop == "depth":
                    # Use negative MSE so higher = better (or use spearman_r)
                    vals.append(result["spearman_r"])
                else:
                    vals.append(result["accuracy"])
            layer_means.append(float(np.mean(vals)) if vals else 0.0)
        summary[prop] = layer_means
    return summary


def _compute_pairwise_layer_summary(per_head_pairwise, n_layers, n_heads):
    """Aggregate pairwise probing results by layer."""
    summary = {}
    # arc -> F1, relation -> accuracy
    metrics = {
        "arc": "f1",
        "relation": "accuracy",
    }
    for task, metric_key in metrics.items():
        layer_means = []
        for l in range(n_layers):
            vals = []
            for h in range(n_heads):
                result = per_head_pairwise.get((l, h), {}).get(task)
                if result is not None and metric_key in result:
                    vals.append(result[metric_key])
            layer_means.append(float(np.mean(vals)) if vals else 0.0)
        summary[task] = layer_means
    return summary


# ---------------------------------------------------------------------------
# Divergence analysis (translator vs impossible)
# ---------------------------------------------------------------------------

def compute_probing_divergence(results_translator, results_impossible, n_layers, n_heads):
    divergence = {}

    # Token-level divergence
    for prop in ["pos", "dep_rel", "depth"]:
        delta = np.zeros((n_layers, n_heads))
        for l in range(n_layers):
            for h in range(n_heads):
                res_t = results_translator["per_head"].get((l, h), {}).get(prop)
                res_i = results_impossible["per_head"].get((l, h), {}).get(prop)
                if res_t is None or res_i is None:
                    delta[l, h] = 0.0
                    continue
                if prop == "depth":
                    delta[l, h] = res_t["spearman_r"] - res_i["spearman_r"]
                else:
                    delta[l, h] = res_t["accuracy"] - res_i["accuracy"]
        divergence[prop] = delta

    # Pairwise divergence
    pw_metrics = {"arc": "f1", "relation": "accuracy"}
    for task, metric_key in pw_metrics.items():
        delta = np.zeros((n_layers, n_heads))
        for l in range(n_layers):
            for h in range(n_heads):
                res_t = results_translator.get("per_head_pairwise", {}).get((l, h), {}).get(task)
                res_i = results_impossible.get("per_head_pairwise", {}).get((l, h), {}).get(task)
                if res_t is None or res_i is None:
                    delta[l, h] = 0.0
                    continue
                delta[l, h] = res_t.get(metric_key, 0.0) - res_i.get(metric_key, 0.0)
        divergence[f"pairwise_{task}"] = delta

    return divergence


def validate_entropy_correlation(
    entropy_divergence: np.ndarray,
    probing_divergence: dict,
):
    flat_entropy = entropy_divergence.flatten()
    correlations = {}
    for prop, delta_acc in probing_divergence.items():
        flat_acc = delta_acc.flatten()
        rho, p_val = stats.spearmanr(flat_entropy, flat_acc)
        correlations[prop] = {
            "rho": float(rho),
            "p_value": float(p_val),
            "significant": bool(abs(rho) > 0.4 and p_val < 0.001),
        }
    return correlations


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_probing_results(results, model_label):
    print(f"\n{'='*60}")
    print(f"  Probing Results: {model_label}")
    print(f"{'='*60}")

    # Word-embedding baseline
    emb = results["word_embedding_baseline"]
    print(f"\n  Word-Embedding Baseline:")
    if emb["pos"]:
        print(f"    POS accuracy:     {emb['pos']['accuracy']:.3f}")
    if emb["dep_rel"]:
        print(f"    Dep-rel accuracy: {emb['dep_rel']['accuracy']:.3f}")
    if emb["depth"]:
        print(f"    Depth spearman r: {emb['depth']['spearman_r']:.3f}")

    # Layer-wise summary
    print(f"\n  Layer-wise mean accuracy / spearman_r:")
    layer_summ = results["layer_summary"]
    for l in range(len(layer_summ["pos"])):
        pos_v = layer_summ["pos"][l]
        dep_v = layer_summ["dep_rel"][l]
        depth_v = layer_summ["depth"][l]
        print(f"    Layer {l:2d}:  POS={pos_v:.3f}  Dep={dep_v:.3f}  Depth={depth_v:.3f}")

    # Top heads per property (token-level)
    for prop in ["pos", "dep_rel", "depth"]:
        metric = "accuracy" if prop != "depth" else "spearman_r"
        scored = []
        for (l, h), res in results["per_head"].items():
            if res[prop] is not None:
                scored.append((l, h, res[prop][metric]))
        scored.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Top-5 heads for {prop} ({metric}):")
        for l, h, v in scored[:5]:
            print(f"    Layer {l}, Head {h}: {v:.3f}")

    # Pairwise results
    pw = results.get("per_head_pairwise", {})
    if pw:
        pw_baselines = results.get("pairwise_baselines", {})
        print(f"\n  Pairwise Baselines:")
        if pw_baselines.get("word_emb_arc"):
            print(f"    Word-emb arc F1:      {pw_baselines['word_emb_arc']['f1']:.3f}")
        if pw_baselines.get("word_emb_rel"):
            print(f"    Word-emb rel acc:     {pw_baselines['word_emb_rel']['accuracy']:.3f}")
        if pw_baselines.get("distance_arc"):
            print(f"    Distance arc F1:      {pw_baselines['distance_arc']['f1']:.3f}")

        pw_summary = results.get("layer_summary_pairwise", {})
        if pw_summary:
            print(f"\n  Layer-wise pairwise (arc F1 / rel acc):")
            n = len(pw_summary.get("arc", []))
            for l in range(n):
                arc_v = pw_summary["arc"][l]
                rel_v = pw_summary["relation"][l]
                print(f"    Layer {l:2d}:  Arc={arc_v:.3f}  Rel={rel_v:.3f}")

        for task, metric in [("arc", "f1"), ("relation", "accuracy")]:
            scored = []
            for (l, h), res in pw.items():
                if res[task] is not None and metric in res[task]:
                    scored.append((l, h, res[task][metric]))
            scored.sort(key=lambda x: x[2], reverse=True)
            print(f"\n  Top-5 heads for pairwise {task} ({metric}):")
            for l, h, v in scored[:5]:
                print(f"    Layer {l}, Head {h}: {v:.3f}")


def print_divergence_results(divergence, correlations):
    print(f"\n{'='*60}")
    print(f"  Probing Divergence (Translator - Impossible)")
    print(f"{'='*60}")
    metric_names = {
        "pos": "ΔAccuracy", "dep_rel": "ΔAccuracy", "depth": "ΔSpearman_r",
        "pairwise_arc": "ΔF1", "pairwise_relation": "ΔAccuracy",
    }
    for prop, delta in divergence.items():
        metric = metric_names.get(prop, "Δmetric")
        print(f"\n  {prop} ({metric}):")
        print(f"    Mean:  {np.mean(delta):.4f}")
        print(f"    Max:   {np.max(delta):.4f}")
        print(f"    Min:   {np.min(delta):.4f}")

        # Top divergent heads
        flat = [(l, h, delta[l, h]) for l in range(delta.shape[0]) for h in range(delta.shape[1])]
        flat.sort(key=lambda x: x[2], reverse=True)
        print(f"    Top-5 positive divergence heads:")
        for l, h, v in flat[:5]:
            print(f"      Layer {l}, Head {h}: {v:+.4f}")

    print(f"\n  Entropy-Probing Correlation (Eq. 7):")
    for prop, c in correlations.items():
        sig = "***" if c["significant"] else "n.s."
        print(f"    {prop}: rho={c['rho']:.3f}, p={c['p_value']:.2e} {sig}")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_results(results):
    """Convert per_head dict with tuple keys to JSON-serialisable format."""
    out = {
        "layer_summary": results["layer_summary"],
        "word_embedding_baseline": results["word_embedding_baseline"],
        "per_head": {},
    }
    for (l, h), v in results["per_head"].items():
        out["per_head"][f"L{l}_H{h}"] = v

    # Pairwise results
    if "per_head_pairwise" in results:
        out["per_head_pairwise"] = {}
        for (l, h), v in results["per_head_pairwise"].items():
            out["per_head_pairwise"][f"L{l}_H{h}"] = v
        out["layer_summary_pairwise"] = results.get("layer_summary_pairwise", {})
        out["pairwise_baselines"] = results.get("pairwise_baselines", {})

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_sentence_split(n_sentences: int, test_size: float = 0.2,
                          random_state: int = 42):
    """Sentence-level train/test split. All tokens of a given sentence fall
    on the same side of the split, eliminating train/test leakage."""
    ids = np.arange(n_sentences)
    if n_sentences < 3:
        return {"train": ids, "test": np.array([], dtype=int)}
    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=random_state, shuffle=True,
    )
    return {"train": np.asarray(train_ids), "test": np.asarray(test_ids)}


def load_dataset(dataset_path, max_sentences=None):
    with open(dataset_path, "r") as f:
        pairs = json.load(f)
    if max_sentences:
        pairs = pairs[:max_sentences]
    scrambled = [pair[0] for pair in pairs]
    original = [pair[1] for pair in pairs]
    return scrambled, original


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probing classifier: test whether attention heads encode syntactic structure"
    )
    parser.add_argument(
        "--translation_model", type=str, required=True,
        help="Path or HF ID for the translator model",
    )
    parser.add_argument(
        "--impossible_model", type=str, required=True,
        help="Path or HF ID for Kallini's impossible model",
    )
    parser.add_argument(
        "--base_model", type=str, default="gpt2",
        help="Path or HF ID for the base GPT-2 model (default: gpt2)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Shared tokenizer (default: use impossible model's tokenizer)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to JSON dataset: [[scrambled, original], ...]",
    )
    parser.add_argument(
        "--entropy_results", type=str, default=None,
        help="Path to entropy results JSON (from attention_entropy.py) for correlation analysis",
    )
    parser.add_argument(
        "--output", type=str, default="probing_results.json",
    )
    parser.add_argument(
        "--max_sentences", type=int, default=None,
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--spacy_model", type=str, default="en_core_web_sm",
        help="spaCy model for syntactic parsing (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--pairwise_max_sentences", type=int, default=None,
        help="Optional sentence cap for pairwise probing only. Useful to avoid OOM.",
    )
    parser.add_argument(
        "--skip_pairwise", action="store_true",
        help="Skip pairwise arc/relation probing entirely.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for transformer/embedding extraction stages (default: 8).",
    )
    parser.add_argument(
        "--pairwise_head_chunk_size", type=int, default=PAIRWISE_HEAD_CHUNK_SIZE,
        help="How many pairwise heads to cache/process at once (default: 1). Lower uses less RAM.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device) if args.device else None
    tokenizer_path = args.tokenizer or args.impossible_model

    # Load dataset
    scrambled_sentences, original_sentences = load_dataset(
        args.dataset, args.max_sentences
    )
    print(f"Loaded {len(scrambled_sentences)} sentence pairs")

    # Build a sentence-level train/test split used by ALL probes (probes.tex
    # §3.3: "same train/test split as the main translation experiments").
    sentence_split = build_sentence_split(len(scrambled_sentences))

    # Syntactic labeler
    labeler = SyntacticLabeler(args.spacy_model)

    # ---- Probe translator model ----
    results_translator = run_probing_pipeline(
        model_path=args.translation_model,
        tokenizer_path=tokenizer_path,
        scrambled_sentences=scrambled_sentences,
        original_sentences=original_sentences,
        labeler=labeler,
        model_label="Translator",
        max_sentences=args.max_sentences,
        device=device,
        sentence_split=sentence_split,
        pairwise_max_sentences=args.pairwise_max_sentences,
        skip_pairwise=args.skip_pairwise,
        batch_size=args.batch_size,
        pairwise_head_chunk_size=args.pairwise_head_chunk_size,
    )
    print_probing_results(results_translator, "Translator")

    # ---- Probe impossible model ----
    results_impossible = run_probing_pipeline(
        model_path=args.impossible_model,
        tokenizer_path=tokenizer_path,
        scrambled_sentences=scrambled_sentences,
        original_sentences=original_sentences,
        labeler=labeler,
        model_label="Impossible",
        max_sentences=args.max_sentences,
        device=device,
        sentence_split=sentence_split,
        pairwise_max_sentences=args.pairwise_max_sentences,
        skip_pairwise=args.skip_pairwise,
        batch_size=args.batch_size,
        pairwise_head_chunk_size=args.pairwise_head_chunk_size,
    )
    print_probing_results(results_impossible, "Impossible")

    # ---- Probe base GPT-2 model ----
    results_base = run_probing_pipeline(
        model_path=args.base_model,
        tokenizer_path=tokenizer_path,
        scrambled_sentences=scrambled_sentences,
        original_sentences=original_sentences,
        labeler=labeler,
        model_label="GPT-2 Base",
        max_sentences=args.max_sentences,
        device=device,
        sentence_split=sentence_split,
        pairwise_max_sentences=args.pairwise_max_sentences,
        skip_pairwise=args.skip_pairwise,
        batch_size=args.batch_size,
        pairwise_head_chunk_size=args.pairwise_head_chunk_size,
    )
    print_probing_results(results_base, "GPT-2 Base")

    # ---- Divergence analysis ----
    # Determine grid size from first model
    sample_key = next(iter(results_translator["per_head"]))
    n_layers = max(k[0] for k in results_translator["per_head"]) + 1
    n_heads = max(k[1] for k in results_translator["per_head"]) + 1

    # Translator vs Impossible
    divergence = compute_probing_divergence(
        results_translator, results_impossible, n_layers, n_heads
    )
    # Translator vs Base
    divergence_vs_base = compute_probing_divergence(
        results_translator, results_base, n_layers, n_heads
    )
    # Base vs Impossible
    divergence_base_vs_impossible = compute_probing_divergence(
        results_base, results_impossible, n_layers, n_heads
    )

    # ---- Entropy correlation (if provided) ----
    correlations = {}
    if args.entropy_results:
        print("\n  Loading entropy results for correlation analysis...")
        with open(args.entropy_results, "r") as f:
            entropy_data = json.load(f)

        # delta_H = H_impossible - H_translator (translation_vs_impossible)
        if "comparisons" in entropy_data:
            delta_H = np.array(
                entropy_data["comparisons"]["translation_vs_impossible"]["delta_H"]
            )
        else:
            # Compute from raw entropy
            H_t = np.array(entropy_data["raw_entropy"]["H_translation"])
            H_i = np.array(entropy_data["raw_entropy"]["H_impossible"])
            delta_H = H_i - H_t

        correlations = validate_entropy_correlation(delta_H, divergence)
    else:
        correlations = {prop: {"rho": None, "p_value": None, "significant": None}
                        for prop in divergence}

    print_divergence_results(divergence, correlations)

    # ---- Save results ----
    output = {
        "models": {
            "translation": args.translation_model,
            "impossible": args.impossible_model,
            "base": args.base_model,
        },
        "tokenizer": tokenizer_path,
        "n_sentences": len(scrambled_sentences),
        "translator": _serialise_results(results_translator),
        "impossible": _serialise_results(results_impossible),
        "base": _serialise_results(results_base),
        "divergence": {prop: delta.tolist() for prop, delta in divergence.items()},
        "divergence_vs_base": {prop: delta.tolist() for prop, delta in divergence_vs_base.items()},
        "divergence_base_vs_impossible": {prop: delta.tolist() for prop, delta in divergence_base_vs_impossible.items()},
        "entropy_probing_correlation": correlations,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


# python prob_classifier.py   \
# --translation_model the-amirhosein/gutenberg-localShuffle-w3 \
# --impossible_model mission-impossible-lms/local-shuffle-w3-gpt2 \
# --base_model gpt2 \
# --dataset test_data/test_data_1k_gutenberg_localShuffle3.json \
# --entropy_results entropy_impossible_results.json \
# --output probing_results.json \
# --max_sentences 1000 \
# --batch_size 32

import argparse
import inspect
import json
from collections import defaultdict, deque
from types import MethodType

import numpy as np
import spacy
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    precision_score, recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward


def detect_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        try:
            if mps_backend.is_available() and mps_backend.is_built():
                return torch.device("mps")
        except Exception:
            pass

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Syntactic label extraction (gold labels from original/possible sentences)
# ---------------------------------------------------------------------------

class SyntacticLabeler:
    """Extract gold POS tags, dependency relations, and syntactic depth
    from unscrambled (possible) sentences using spaCy."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def _extract_labels_from_doc(self, doc):
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

    def extract_labels(self, sentence: str):
        """Return per-token POS, dep relation, head index, and tree depth."""
        return self._extract_labels_from_doc(self.nlp(sentence))

    def extract_labels_batch(self, sentences: list[str], batch_size: int = 128):
        """Parse many original sentences at once with spaCy.pipe."""
        return [
            self._extract_labels_from_doc(doc)
            for doc in self.nlp.pipe(sentences, batch_size=batch_size)
        ]

    def tokenize_batch(self, sentences: list[str], batch_size: int = 256):
        """Tokenize many scrambled sentences without running the parser."""
        return list(self.nlp.tokenizer.pipe(sentences, batch_size=batch_size))

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

    def __init__(self, model_path: str, tokenizer_path: str = None, device=None, fp16: bool = False):
        if device is None:
            self.device = detect_best_device()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
        print(f"Using device: {self.device}", flush=True)

        tok_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if fp16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
            output_hidden_states=False,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        cfg = self.model.config
        self.n_layers = cfg.n_layer
        self.n_heads = cfg.n_head
        self.d_head = cfg.n_embd // cfg.n_head  # 64 for GPT-2
        self.model_path = model_path

        # Patch attention blocks to capture true per-head outputs
        self._head_outputs = {}
        self._original_attn_forwards = {}
        self._upcast_accepts_head_mask = False
        self._patch_attention_modules()

    # -- attention patching -------------------------------------------------

    def _patch_attention_modules(self):
        """Patch GPT-2 attention blocks to capture outputs before c_proj mixes heads."""
        for layer_idx in range(self.n_layers):
            attn_module = self.model.transformer.h[layer_idx].attn
            if layer_idx == 0:
                upcast_sig = inspect.signature(attn_module._upcast_and_reordered_attn)
                self._upcast_accepts_head_mask = "head_mask" in upcast_sig.parameters
            self._original_attn_forwards[layer_idx] = attn_module.forward
            attn_module.forward = MethodType(
                self._make_patched_forward(layer_idx), attn_module
            )

    def _make_patched_forward(self, layer_idx: int):
        original_forward = self._original_attn_forwards[layer_idx]

        def patched_forward(
            attn_module,
            hidden_states,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            **kwargs,
        ):
            # Fall back to HF's implementation for code paths we do not use
            # in probing extraction (for example cached decoding).
            if encoder_hidden_states is not None or past_key_values is not None:
                return original_forward(
                    hidden_states,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            query_states, key_states, value_states = attn_module.c_attn(hidden_states).split(
                attn_module.split_size, dim=2
            )
            shape_kv = (*key_states.shape[:-1], -1, attn_module.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

            shape_q = (*query_states.shape[:-1], -1, attn_module.head_dim)
            query_states = query_states.view(shape_q).transpose(1, 2)

            is_causal = attention_mask is None and query_states.shape[-2] > 1
            attention_interface = eager_attention_forward
            if attn_module.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn_module.config._attn_implementation]

            if attn_module.config._attn_implementation == "eager" and attn_module.reorder_and_upcast_attn:
                if self._upcast_accepts_head_mask:
                    attn_output, attn_weights = attn_module._upcast_and_reordered_attn(
                        query_states, key_states, value_states, attention_mask, head_mask
                    )
                else:
                    attn_output, attn_weights = attn_module._upcast_and_reordered_attn(
                        query_states, key_states, value_states, attention_mask
                    )
            else:
                attn_output, attn_weights = attention_interface(
                    attn_module,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    head_mask=head_mask,
                    dropout=attn_module.attn_dropout.p if attn_module.training else 0.0,
                    is_causal=is_causal,
                    **kwargs,
                )

            # Save true per-head outputs before c_proj mixes head subspaces.
            # Keep on GPU until batch is done — no per-layer CPU sync.
            self._head_outputs[layer_idx] = attn_output.permute(0, 2, 1, 3).detach()
            # shape: (batch, seq, n_heads, d_head) — stays on GPU until batch is done

            attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
            attn_output = attn_module.c_proj(attn_output)
            attn_output = attn_module.resid_dropout(attn_output)
            return attn_output, attn_weights

        return patched_forward

    def remove_hooks(self):
        for layer_idx in range(self.n_layers):
            attn_module = self.model.transformer.h[layer_idx].attn
            if layer_idx in self._original_attn_forwards:
                attn_module.forward = self._original_attn_forwards[layer_idx]
        self._original_attn_forwards.clear()

    def tokenize(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = [tuple(span) for span in encoded.pop("offset_mapping")[0].tolist()]
        input_ids = encoded["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        model_inputs = {k: v.to(self.device) for k, v in encoded.items()}
        return model_inputs, input_ids, tokens, offsets

    # -- extraction ---------------------------------------------------------

    @torch.inference_mode()
    def extract_batch(self, texts: list[str]) -> list[tuple]:
        """Tokenize all texts at once and run ONE forward pass for the whole batch.

        Returns a list of (head_reps, token_ids, tokens, offsets) tuples,
        one per input text, where head_reps is {(layer_idx, head_idx): np.ndarray
        of shape (seq_len, d_head)}.
        """
        self._head_outputs.clear()

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        # Extract offset mapping before passing to model
        offset_mapping_batch = encoded.pop("offset_mapping")  # (batch, seq)
        attention_mask = encoded["attention_mask"]  # (batch, seq) on CPU still

        model_inputs = {k: v.to(self.device) for k, v in encoded.items()}
        attention_mask_device = model_inputs["attention_mask"]

        self.model(**model_inputs)

        # Bulk CPU transfer — one transfer per layer for the whole batch
        head_outputs_cpu = {
            l: self._head_outputs[l].cpu().float().numpy()
            for l in range(self.n_layers)
        }
        self._head_outputs.clear()

        results = []
        batch_size = len(texts)
        attention_mask_np = attention_mask.numpy()

        for b_idx in range(batch_size):
            seq_len = int(attention_mask_np[b_idx].sum())
            if seq_len == 0:
                results.append((None, [], [], []))
                continue

            # Per-sample token ids and tokens (non-padded portion)
            input_ids_b = model_inputs["input_ids"][b_idx, :seq_len].tolist()
            tokens_b = self.tokenizer.convert_ids_to_tokens(input_ids_b)
            offsets_b = [
                tuple(span)
                for span in offset_mapping_batch[b_idx, :seq_len].tolist()
            ]

            head_reps = {}
            for l in range(self.n_layers):
                per_head = head_outputs_cpu[l][b_idx, :seq_len, :, :]  # (seq, n_heads, d_head)
                for h in range(self.n_heads):
                    arr = per_head[:, h, :]
                    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    head_reps[(l, h)] = arr

            results.append((head_reps, input_ids_b, tokens_b, offsets_b))

        return results

    def extract(self, text: str):
        return self.extract_batch([text])[0]


# ---------------------------------------------------------------------------
# Subword-to-word alignment
# ---------------------------------------------------------------------------

def _token_identity(text: str) -> str:
    return text


def _first_subword_for_span(offsets, start_char: int, end_char: int):
    for subword_idx, (sub_start, sub_end) in enumerate(offsets):
        if sub_end <= start_char:
            continue
        if sub_start >= end_char:
            break
        if max(sub_start, start_char) < min(sub_end, end_char):
            return subword_idx
    return None


def align_scrambled_words_to_labels(
    scrambled_sentence: str,
    offsets,
    labeler: SyntacticLabeler,
    word_labels: dict,
    scrambled_doc=None,
):
    """Map scrambled tokens to original labels by token identity and occurrence."""
    if scrambled_doc is None:
        scrambled_doc = labeler.nlp.make_doc(scrambled_sentence)

    original_occurrences = defaultdict(deque)
    for original_idx, token_text in enumerate(word_labels["tokens"]):
        original_occurrences[_token_identity(token_text)].append(original_idx)

    aligned = []
    for scrambled_idx, tok in enumerate(scrambled_doc):
        identity = _token_identity(tok.text)
        if not original_occurrences[identity]:
            continue

        original_idx = original_occurrences[identity].popleft()
        subword_idx = _first_subword_for_span(offsets, tok.idx, tok.idx + len(tok))
        if subword_idx is None:
            continue

        aligned.append({
            "subword_idx": subword_idx,
            "scrambled_word_idx": scrambled_idx,
            "word_idx": original_idx,
            "token": tok.text,
            "pos": word_labels["pos"][original_idx],
            "dep_rel": word_labels["dep_rel"][original_idx],
            "head_idx": word_labels["head_idx"][original_idx],
            "depth": word_labels["depth"][original_idx],
        })

    return aligned


def build_sentence_split(n_sentences: int, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    sentence_ids = np.arange(n_sentences)
    if n_sentences < 3:
        return {"train": sentence_ids, "val": np.array([], dtype=int), "test": np.array([], dtype=int)}

    train_ids, test_ids = train_test_split(
        sentence_ids, test_size=test_size, random_state=random_state, shuffle=True
    )
    val_fraction = val_size / max(1e-8, 1.0 - test_size)
    if len(train_ids) >= 2 and 0 < val_fraction < 1:
        train_ids, val_ids = train_test_split(
            train_ids, test_size=val_fraction, random_state=random_state, shuffle=True
        )
    else:
        val_ids = np.array([], dtype=int)
    return {"train": np.asarray(train_ids), "val": np.asarray(val_ids), "test": np.asarray(test_ids)}


def prepare_syntax_cache(
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    batch_size: int = 128,
):
    """Precompute reusable CPU-side syntax data shared across model runs."""
    print("\nPrecomputing syntax cache...", flush=True)
    original_labels = labeler.extract_labels_batch(
        original_sentences, batch_size=batch_size,
    )
    scrambled_docs = labeler.tokenize_batch(
        scrambled_sentences, batch_size=max(64, batch_size * 2),
    )
    return {
        "original_labels": original_labels,
        "scrambled_docs": scrambled_docs,
    }


# ---------------------------------------------------------------------------
# Dataset construction for probing
# ---------------------------------------------------------------------------

def build_probing_dataset(
    extractor: HeadRepresentationExtractor,
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    max_sentences: int = None,
    batch_size: int = 16,
    original_labels_list: list[dict] | None = None,
    scrambled_docs: list | None = None,
):
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]
        if original_labels_list is not None:
            original_labels_list = original_labels_list[:max_sentences]
        if scrambled_docs is not None:
            scrambled_docs = scrambled_docs[:max_sentences]

    if original_labels_list is None:
        original_labels_list = labeler.extract_labels_batch(
            original_sentences, batch_size=max(64, batch_size * 4),
        )
    if scrambled_docs is None:
        scrambled_docs = labeler.tokenize_batch(
            scrambled_sentences, batch_size=max(64, batch_size * 8),
        )

    n_layers = extractor.n_layers
    n_heads = extractor.n_heads

    dataset = {
        (l, h): {"X": []}
        for l in range(n_layers)
        for h in range(n_heads)
    }
    shared_pos = []
    shared_dep_rel = []
    shared_depth = []
    shared_head_idx = []
    shared_sentence_id = []

    skipped = 0
    total = len(scrambled_sentences)

    for batch_start in range(0, total, batch_size):
        batch_scrambled = scrambled_sentences[batch_start:batch_start + batch_size]
        batch_original = original_sentences[batch_start:batch_start + batch_size]

        if batch_start % 100 == 0:
            print(f"  Building probing data: {batch_start}/{total}", flush=True)

        batch_results = extractor.extract_batch(batch_scrambled)

        for i_local, (scrambled, original, extraction) in enumerate(
            zip(batch_scrambled, batch_original, batch_results)
        ):
            i = batch_start + i_local
            head_reps, token_ids, tokens, offsets = extraction

            if head_reps is None:
                skipped += 1
                continue

            labels = original_labels_list[i]
            scrambled_doc = scrambled_docs[i]

            # Align scrambled words back to original labels by identity.
            aligned = align_scrambled_words_to_labels(
                scrambled, offsets, labeler, labels, scrambled_doc=scrambled_doc,
            )
            if len(aligned) == 0:
                skipped += 1
                continue

            seq_len = next(iter(head_reps.values())).shape[0]
            valid_entries = [
                entry for entry in aligned
                if entry["subword_idx"] < seq_len
            ]
            if not valid_entries:
                skipped += 1
                continue

            sw_indices = np.array(
                [entry["subword_idx"] for entry in valid_entries],
                dtype=np.int32,
            )

            shared_pos.extend(entry["pos"] for entry in valid_entries)
            shared_dep_rel.extend(entry["dep_rel"] for entry in valid_entries)
            shared_depth.extend(entry["depth"] for entry in valid_entries)
            shared_head_idx.extend(entry["head_idx"] for entry in valid_entries)
            shared_sentence_id.extend([i] * len(valid_entries))

            for key, rep in head_reps.items():
                dataset[key]["X"].append(rep[sw_indices])

    if skipped:
        print(
            f"  Skipped {skipped}/{total} sentences (alignment)",
            flush=True,
        )

    shared_depth_arr = np.array(shared_depth, dtype=np.float32)
    shared_sentence_id_arr = np.array(shared_sentence_id, dtype=np.int32)

    # Convert lists to arrays
    for key in dataset:
        if len(dataset[key]["X"]) > 0:
            dataset[key]["X"] = np.concatenate(dataset[key]["X"], axis=0)
        else:
            dataset[key]["X"] = np.empty((0, extractor.d_head))
        dataset[key]["pos"] = shared_pos
        dataset[key]["dep_rel"] = shared_dep_rel
        dataset[key]["depth"] = shared_depth_arr
        dataset[key]["head_idx"] = shared_head_idx
        dataset[key]["sentence_id"] = shared_sentence_id_arr

    return dataset


# ---------------------------------------------------------------------------
# Probing classifiers
# ---------------------------------------------------------------------------

class ProbingExperiment:
    """Train and evaluate linear probes for POS, dependency, and depth."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42, sentence_split=None):
        self.test_size = test_size
        self.random_state = random_state
        self.sentence_split = sentence_split

    # -- NaN sanitiser ------------------------------------------------------

    @staticmethod
    def _sanitise(X: np.ndarray, y: np.ndarray, sentence_ids: np.ndarray | None = None):
        """Drop rows where any feature is NaN or Inf.

        Returns (X_clean, y_clean) or (None, None) if too few rows survive.
        """
        finite_mask = np.isfinite(X).all(axis=1)
        n_bad = (~finite_mask).sum()
        if n_bad:
            print(f"    [sanitise] dropping {n_bad} rows with NaN/Inf features", flush=True)
        X_c, y_c = X[finite_mask], y[finite_mask]
        sent_c = sentence_ids[finite_mask] if sentence_ids is not None else None
        if len(X_c) < 20:
            return None, None, None
        return X_c, y_c, sent_c

    # -- Safe split helper --------------------------------------------------

    def _safe_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        sentence_ids: np.ndarray | None = None,
    ):
        """Stratified split when possible; drops singleton classes otherwise.

        Returns (X_train, X_test, y_train, y_test) or None if too few samples
        remain after filtering rare classes.
        """
        if sentence_ids is not None and self.sentence_split is not None:
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

    def probe_pos(self, X: np.ndarray, labels: list[str], sentence_ids: np.ndarray | None = None):
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

        class_counts = np.bincount(y)
        rare_classes = np.where(class_counts < 2)[0]
        if len(rare_classes) > 0:
            keep_mask = np.isin(y, rare_classes, invert=True)
            X, y = X[keep_mask], y[keep_mask]
            sentence_ids = sentence_ids[keep_mask] if sentence_ids is not None else None
            if len(X) < 20:
                return None

        n_classes = len(np.unique(y))

        split = self._safe_train_test_split(X, y, n_classes, sentence_ids)
        if split is None:
            return None
        X_train, X_test, y_train, y_test = split

        # Linear probe
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Baselines
        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        # Random label baseline
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        clf_rand = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf_rand.fit(X_train, y_shuffled)
        rand_label_acc = accuracy_score(y_test, clf_rand.predict(X_test))

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

    def probe_dependency(self, X: np.ndarray, labels: list[str], sentence_ids: np.ndarray | None = None):
        """Linear probe for dependency relation classification."""
        if len(X) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X, y, sentence_ids = self._sanitise(X, y, sentence_ids)
        if X is None:
            return None

        class_counts = np.bincount(y)
        rare_classes = np.where(class_counts < 2)[0]
        if len(rare_classes) > 0:
            keep_mask = np.isin(y, rare_classes, invert=True)
            X, y = X[keep_mask], y[keep_mask]
            sentence_ids = sentence_ids[keep_mask] if sentence_ids is not None else None
            if len(X) < 20:
                return None

        n_classes = len(np.unique(y))

        split = self._safe_train_test_split(X, y, n_classes, sentence_ids)
        if split is None:
            return None
        X_train, X_test, y_train, y_test = split

        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        clf_rand = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf_rand.fit(X_train, y_shuffled)
        rand_label_acc = accuracy_score(y_test, clf_rand.predict(X_test))

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

    def probe_depth(self, X: np.ndarray, depths: np.ndarray, sentence_ids: np.ndarray | None = None):
        """Linear probe for syntactic depth (regression)."""
        if len(X) < 20:
            return None

        X, depths, sentence_ids = self._sanitise(X, depths, sentence_ids)
        if X is None:
            return None

        if sentence_ids is not None and self.sentence_split is not None:
            train_mask = np.isin(sentence_ids, self.sentence_split["train"])
            test_mask = np.isin(sentence_ids, self.sentence_split["test"])
            if not train_mask.any() or not test_mask.any():
                return None
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = depths[train_mask], depths[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, depths, test_size=self.test_size, random_state=self.random_state,
            )

        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="auto"))
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

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
        reg_rand = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="auto"))
        reg_rand.fit(X_train, y_shuffled)
        rand_mse = mean_squared_error(y_test, reg_rand.predict(X_test))

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
    ):
        assert combination in self.COMBINATION_METHODS
        self.combination = combination
        self.test_size = test_size
        self.random_state = random_state
        self.max_pairs_per_sentence = max_pairs_per_sentence
        self.sentence_split = sentence_split

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
            print(f"    [pairwise sanitise] dropping {n_bad} rows with NaN/Inf", flush=True)
        X_c, y_c = X[finite_mask], y[finite_mask]
        sent_c = sentence_ids[finite_mask] if sentence_ids is not None else None
        if len(X_c) < 20:
            return None, None, None
        return X_c, y_c, sent_c

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

        if sentence_ids is not None and self.sentence_split is not None:
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

        # Class-weighted logistic regression for sparse arcs
        weights = compute_class_weight(
            "balanced", classes=np.array([0, 1]), y=y_train,
        )
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            class_weight={0: weights[0], 1: weights[1]},
            random_state=self.random_state,         ))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

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
        clf_rand = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf_rand.fit(X_train, y_shuffled)
        rand_f1 = f1_score(y_test, clf_rand.predict(X_test), zero_division=0)

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

        # Drop singleton classes for stratified split
        class_counts = np.bincount(y)
        rare = np.where(class_counts < 2)[0]
        if len(rare) > 0:
            mask = np.isin(y, rare, invert=True)
            X_pairs, y = X_pairs[mask], y[mask]
            sentence_ids = sentence_ids[mask] if sentence_ids is not None else None
            if len(X_pairs) < 20:
                return None
            n_classes = len(np.unique(y))

        if sentence_ids is not None and self.sentence_split is not None:
            train_mask = np.isin(sentence_ids, self.sentence_split["train"])
            test_mask = np.isin(sentence_ids, self.sentence_split["test"])
            if not train_mask.any() or not test_mask.any():
                return None
            X_train, X_test = X_pairs[train_mask], X_pairs[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 1:
                return None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_pairs, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

        clf = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        clf_rand = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, solver="lbfgs",
            random_state=self.random_state,         ))
        clf_rand.fit(X_train, y_shuffled)
        rand_label_acc = accuracy_score(y_test, clf_rand.predict(X_test))

        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "n_classes": n_classes,
            "n_samples": len(X_pairs),
            "classes": le.inverse_transform(np.unique(y)).tolist(),
            "random_baseline": float(random_acc),
            "majority_baseline": float(majority_acc),
            "random_label_baseline": float(rand_label_acc),
        }



# ---------------------------------------------------------------------------
# Pairwise dataset construction
# ---------------------------------------------------------------------------

def build_pairwise_dataset(
    extractor: HeadRepresentationExtractor,
    labeler: SyntacticLabeler,
    scrambled_sentences: list[str],
    original_sentences: list[str],
    combination: str = "concat",
    max_sentences: int = None,
    batch_size: int = 16,
    original_labels_list: list[dict] | None = None,
    scrambled_docs: list | None = None,
):
    """Build pairwise token-pair dataset for dependency structure probing.

    For each sentence, constructs all ordered pairs (i, j) where i != j,
    with gold labels from the original parse.

    Returns dict keyed by (layer, head) with:
        X_pairs: combined representations c_ij
        y_arc: binary arc labels
        y_rel: relation labels (for positive pairs)
        distances: |i - j| word distance for distance baseline
    """
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]
        if original_labels_list is not None:
            original_labels_list = original_labels_list[:max_sentences]
        if scrambled_docs is not None:
            scrambled_docs = scrambled_docs[:max_sentences]

    if original_labels_list is None:
        original_labels_list = labeler.extract_labels_batch(
            original_sentences, batch_size=max(64, batch_size * 4),
        )
    if scrambled_docs is None:
        scrambled_docs = labeler.tokenize_batch(
            scrambled_sentences, batch_size=max(64, batch_size * 8),
        )

    n_layers = extractor.n_layers
    n_heads = extractor.n_heads

    # Collect per-sentence pairwise data, then combine per-head
    # To save memory, accumulate X for one (layer, head) at a time is impractical
    # Instead, collect aligned sentence data first, then build pairs per head

    sentence_data = []  # list of dicts with aligned info and head_reps

    skipped = 0
    total = len(scrambled_sentences)

    for batch_start in range(0, total, batch_size):
        batch_scrambled = scrambled_sentences[batch_start:batch_start + batch_size]
        batch_original = original_sentences[batch_start:batch_start + batch_size]

        if batch_start % 100 == 0:
            print(f"  Building pairwise data: {batch_start}/{total}", flush=True)

        batch_results = extractor.extract_batch(batch_scrambled)

        for i_local, (scrambled, original, extraction) in enumerate(
            zip(batch_scrambled, batch_original, batch_results)
        ):
            i = batch_start + i_local
            head_reps, token_ids, tokens, offsets = extraction

            if head_reps is None:
                skipped += 1
                continue

            labels = original_labels_list[i]
            aligned = align_scrambled_words_to_labels(
                scrambled, offsets, labeler, labels, scrambled_doc=scrambled_docs[i],
            )
            if len(aligned) < 2:  # need at least 2 tokens for pairs
                skipped += 1
                continue

            sentence_data.append({
                "aligned": aligned,
                "head_reps": head_reps,
                "n_aligned": len(aligned),
                "sentence_id": i,
            })

    if skipped:
        print(
            f"  Skipped {skipped}/{total} sentences (pairwise alignment)",
            flush=True,
        )

    # Now build per-head pairwise datasets using numpy vectorization
    # Accumulators keyed by (layer, head)
    accumulators = {
        (l, h): {
            "X_pairs": [],
            "y_arc": [],
            "X_pairs_pos": [],
            "y_rel_pos": [],
            "distances": [],
            "sentence_ids": [],
            "sentence_ids_pos": [],
        }
        for l in range(n_layers)
        for h in range(n_heads)
    }

    for sent in sentence_data:
        aligned = sent["aligned"]
        n_tok = sent["n_aligned"]

        # Build all_reps: (n_layers, n_heads, seq, d_head)
        all_reps = np.stack([
            np.stack([sent["head_reps"][(l, h)] for h in range(n_heads)])
            for l in range(n_layers)
        ])

        sw_indices = np.array([a["subword_idx"] for a in aligned])
        valid = sw_indices < all_reps.shape[2]
        sw_indices = sw_indices[valid]
        aligned_f = [a for a, v in zip(aligned, valid) if v]
        n_tok_v = len(aligned_f)

        if n_tok_v < 2:
            continue

        # indexed_reps shape: (n_layers, n_heads, n_tok_v, d_head)
        indexed_reps = all_reps[:, :, sw_indices, :]

        # Build all i!=j pairs with meshgrid
        ii, jj = np.meshgrid(np.arange(n_tok_v), np.arange(n_tok_v), indexing='ij')
        pair_mask = ii != jj
        i_pairs = ii[pair_mask]
        j_pairs = jj[pair_mask]
        n_pairs = len(i_pairs)

        # h_i_all, h_j_all shape: (n_layers, n_heads, n_pairs, d_head)
        h_i_all = indexed_reps[:, :, i_pairs, :]
        h_j_all = indexed_reps[:, :, j_pairs, :]

        # For "concat" combination (the only one used in run_probing_pipeline):
        if combination == "concat":
            c_ij_all = np.concatenate([h_i_all, h_j_all], axis=-1)  # (n_layers, n_heads, n_pairs, 2*d_head)
        elif combination == "diff":
            c_ij_all = h_i_all - h_j_all
        elif combination == "product":
            c_ij_all = h_i_all * h_j_all
        else:  # "full"
            c_ij_all = np.concatenate([h_i_all, h_j_all, h_i_all - h_j_all, h_i_all * h_j_all], axis=-1)

        # Arc labels (same for all heads)
        arc_labels = []
        rel_labels = []
        distances = []
        pos_pair_mask = []

        for k in range(n_pairs):
            pi = i_pairs[k]
            pj = j_pairs[k]
            head_of_i = aligned_f[pi]["head_idx"]
            word_idx_j = aligned_f[pj].get("word_idx", int(pj))
            is_arc = int(head_of_i == word_idx_j)
            arc_labels.append(is_arc)
            rel_labels.append(aligned_f[pi]["dep_rel"] if is_arc else "NO_ARC")
            distances.append(abs(
                aligned_f[pi]["scrambled_word_idx"] - aligned_f[pj]["scrambled_word_idx"]
            ))
            pos_pair_mask.append(bool(is_arc))

        arc_labels = np.array(arc_labels, dtype=np.int32)
        distances = np.array(distances, dtype=np.int32)
        pos_pair_mask = np.array(pos_pair_mask)
        rel_labels_pos = [rel_labels[k] for k in range(n_pairs) if pos_pair_mask[k]]
        sid = sent["sentence_id"]

        # Append to accumulators for each (l, h)
        for l in range(n_layers):
            for h in range(n_heads):
                pairs_lh = c_ij_all[l, h]  # (n_pairs, feat_dim)
                acc = accumulators[(l, h)]
                acc["X_pairs"].append(pairs_lh)
                acc["y_arc"].append(arc_labels)
                acc["distances"].append(distances)
                acc["sentence_ids"].append(np.full(n_pairs, sid, dtype=np.int32))
                if pos_pair_mask.any():
                    acc["X_pairs_pos"].append(pairs_lh[pos_pair_mask])
                    acc["y_rel_pos"].extend(rel_labels_pos)
                    acc["sentence_ids_pos"].append(
                        np.full(pos_pair_mask.sum(), sid, dtype=np.int32)
                    )

    # Assemble final dataset from accumulators
    dataset = {}
    d_comb = extractor.d_head * (2 if combination == "concat" else
                                  1 if combination in ("diff", "product") else 4)

    for l in range(n_layers):
        for h in range(n_heads):
            acc = accumulators[(l, h)]
            if len(acc["X_pairs"]) > 0:
                dataset[(l, h)] = {
                    "X_pairs": np.concatenate(acc["X_pairs"], axis=0),
                    "y_arc": np.concatenate(acc["y_arc"], axis=0),
                    "X_pairs_pos": (
                        np.concatenate(acc["X_pairs_pos"], axis=0)
                        if acc["X_pairs_pos"]
                        else np.empty((0, d_comb))
                    ),
                    "y_rel_pos": acc["y_rel_pos"],
                    "distances": np.concatenate(acc["distances"], axis=0),
                    "sentence_id": np.concatenate(acc["sentence_ids"], axis=0),
                    "sentence_id_pos": (
                        np.concatenate(acc["sentence_ids_pos"], axis=0)
                        if acc["sentence_ids_pos"]
                        else np.empty(0, dtype=np.int32)
                    ),
                }
            else:
                dataset[(l, h)] = {
                    "X_pairs": np.empty((0, d_comb)),
                    "y_arc": np.empty(0, dtype=np.int32),
                    "X_pairs_pos": np.empty((0, d_comb)),
                    "y_rel_pos": [],
                    "distances": np.empty(0, dtype=np.int32),
                    "sentence_id": np.empty(0, dtype=np.int32),
                    "sentence_id_pos": np.empty(0, dtype=np.int32),
                }

    return dataset


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
    batch_size: int = 16,
    original_labels_list: list[dict] | None = None,
    scrambled_docs: list | None = None,
):
    """Word-embedding pairwise baseline and distance baseline."""
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]
        if original_labels_list is not None:
            original_labels_list = original_labels_list[:max_sentences]
        if scrambled_docs is not None:
            scrambled_docs = scrambled_docs[:max_sentences]

    if original_labels_list is None:
        original_labels_list = labeler.extract_labels_batch(
            original_sentences, batch_size=max(64, batch_size * 4),
        )
    if scrambled_docs is None:
        scrambled_docs = labeler.tokenize_batch(
            scrambled_sentences, batch_size=max(64, batch_size * 8),
        )

    combiner = PairwiseDependencyProber(combination=combination)
    wte = extractor.model.transformer.wte

    X_emb_pairs = []
    y_arc_all = []
    y_rel_pos = []
    X_emb_pos = []
    distances_all = []
    sentence_ids_all = []
    sentence_ids_pos = []

    total = len(scrambled_sentences)
    for i, (scrambled, original) in enumerate(
        zip(scrambled_sentences, original_sentences)
    ):
        if i % 100 == 0:
            print(f"  Pairwise baseline: {i}/{total}", flush=True)

        labels = original_labels_list[i]
        inputs, input_ids, tokens, offsets = extractor.tokenize(scrambled)

        if inputs["input_ids"].shape[1] == 0:
            continue

        with torch.no_grad():
            embeddings = wte(inputs["input_ids"])[0].cpu().numpy()

        aligned = align_scrambled_words_to_labels(
            scrambled, offsets, labeler, labels, scrambled_doc=scrambled_docs[i],
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
                is_arc = 1 if head_of_i == aligned[idx_j].get("word_idx", idx_j) else 0
                rel_label = aligned[idx_i]["dep_rel"] if is_arc else "NO_ARC"

                X_emb_pairs.append(c_ij)
                y_arc_all.append(is_arc)
                distances_all.append(
                    abs(
                        aligned[idx_i]["scrambled_word_idx"]
                        - aligned[idx_j]["scrambled_word_idx"]
                    )
                )
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
    )

    # Word-embedding arc baseline
    result["word_emb_arc"] = prober.probe_arc(X_emb_pairs, y_arc_all, sentence_ids_all)

    # Word-embedding relation baseline (positive pairs only)
    if len(X_emb_pos) >= 20:
        X_emb_pos = np.stack(X_emb_pos)
        result["word_emb_rel"] = prober.probe_relation(X_emb_pos, y_rel_pos, sentence_ids_pos)

    # Distance baseline: predict arc from |i-j| alone
    dist_features = distances_all.reshape(-1, 1).astype(np.float32)
    result["distance_arc"] = prober.probe_arc(dist_features, y_arc_all, sentence_ids_all)

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
    batch_size: int = 16,
    original_labels_list: list[dict] | None = None,
    scrambled_docs: list | None = None,
):
    """Train probes on word embeddings (wte) as a control baseline."""
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]
        if original_labels_list is not None:
            original_labels_list = original_labels_list[:max_sentences]
        if scrambled_docs is not None:
            scrambled_docs = scrambled_docs[:max_sentences]

    if original_labels_list is None:
        original_labels_list = labeler.extract_labels_batch(
            original_sentences, batch_size=max(64, batch_size * 4),
        )
    if scrambled_docs is None:
        scrambled_docs = labeler.tokenize_batch(
            scrambled_sentences, batch_size=max(64, batch_size * 8),
        )

    X_all, pos_all, dep_all, depth_all, sentence_ids = [], [], [], [], []
    wte = extractor.model.transformer.wte  # word token embedding layer

    for i, (scrambled, original) in enumerate(
        zip(scrambled_sentences, original_sentences)
    ):
        if i % 100 == 0:
            print(f"  Word-embedding baseline: {i}/{len(scrambled_sentences)}", flush=True)

        labels = original_labels_list[i]
        inputs, input_ids, tokens, offsets = extractor.tokenize(scrambled)

        if inputs["input_ids"].shape[1] == 0:
            continue

        with torch.no_grad():
            embeddings = wte(inputs["input_ids"])[0].cpu().numpy()  # (seq, emb)

        aligned = align_scrambled_words_to_labels(
            scrambled, offsets, labeler, labels, scrambled_doc=scrambled_docs[i],
        )

        for entry in aligned:
            sw_idx = entry["subword_idx"]
            if sw_idx < embeddings.shape[0]:
                X_all.append(embeddings[sw_idx])
                pos_all.append(entry["pos"])
                dep_all.append(entry["dep_rel"])
                depth_all.append(entry["depth"])
                sentence_ids.append(i)

    if len(X_all) < 20:
        return {"pos": None, "dep_rel": None, "depth": None}

    X_all = np.stack(X_all)
    depth_all = np.array(depth_all, dtype=np.float32)
    sentence_ids = np.array(sentence_ids, dtype=np.int32)

    prober = ProbingExperiment(
        test_size=test_size,
        random_state=random_state,
        sentence_split=sentence_split,
    )
    return {
        "pos": prober.probe_pos(X_all, pos_all, sentence_ids),
        "dep_rel": prober.probe_dependency(X_all, dep_all, sentence_ids),
        "depth": prober.probe_depth(X_all, depth_all, sentence_ids),
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
    batch_size: int = 16,
    fp16: bool = False,
    syntax_cache: dict | None = None,
):

    print(f"\n{'='*60}", flush=True)
    print(f"  Probing: {model_label} ({model_path})", flush=True)
    print(f"{'='*60}", flush=True)

    extractor = HeadRepresentationExtractor(
        model_path, tokenizer_path=tokenizer_path, device=device, fp16=fp16,
    )

    # Build dataset
    print("\n  Building probing dataset...", flush=True)
    dataset = build_probing_dataset(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
        batch_size=batch_size,
        original_labels_list=(
            syntax_cache["original_labels"] if syntax_cache is not None else None
        ),
        scrambled_docs=(
            syntax_cache["scrambled_docs"] if syntax_cache is not None else None
        ),
    )

    # Word-embedding baseline (token-level)
    print("\n  Computing word-embedding baseline...", flush=True)
    emb_baseline = compute_word_embedding_baseline(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
        sentence_split=sentence_split,
        batch_size=batch_size,
        original_labels_list=(
            syntax_cache["original_labels"] if syntax_cache is not None else None
        ),
        scrambled_docs=(
            syntax_cache["scrambled_docs"] if syntax_cache is not None else None
        ),
    )

    # Probe each head (token-level)
    prober = ProbingExperiment(sentence_split=sentence_split)
    n_layers = extractor.n_layers
    n_heads = extractor.n_heads

    per_head = {}
    total = n_layers * n_heads
    done = 0

    for l in range(n_layers):
        for h in range(n_heads):
            done += 1
            if done % 24 == 0:
                print(f"  Probing head {done}/{total}", flush=True)

            data = dataset[(l, h)]
            X = data["X"]
            if X.shape[0] < 20:
                per_head[(l, h)] = {"pos": None, "dep_rel": None, "depth": None}
                continue

            pos_result = prober.probe_pos(X, data["pos"], data["sentence_id"])
            dep_result = prober.probe_dependency(X, data["dep_rel"], data["sentence_id"])
            depth_result = prober.probe_depth(X, data["depth"], data["sentence_id"])

            per_head[(l, h)] = {
                "pos": pos_result,
                "dep_rel": dep_result,
                "depth": depth_result,
            }

    # ---- Pairwise dependency structure probing ----
    print("\n  Building pairwise probing dataset...", flush=True)
    pw_dataset = build_pairwise_dataset(
        extractor, labeler, scrambled_sentences, original_sentences,
        combination="concat",
        max_sentences=max_sentences,
        batch_size=batch_size,
        original_labels_list=(
            syntax_cache["original_labels"] if syntax_cache is not None else None
        ),
        scrambled_docs=(
            syntax_cache["scrambled_docs"] if syntax_cache is not None else None
        ),
    )

    print("\n  Computing pairwise baselines...", flush=True)
    pw_baselines = compute_pairwise_baselines(
        extractor, labeler, scrambled_sentences, original_sentences,
        combination="concat",
        max_sentences=max_sentences,
        sentence_split=sentence_split,
        batch_size=batch_size,
        original_labels_list=(
            syntax_cache["original_labels"] if syntax_cache is not None else None
        ),
        scrambled_docs=(
            syntax_cache["scrambled_docs"] if syntax_cache is not None else None
        ),
    )

    pw_prober = PairwiseDependencyProber(combination="concat", sentence_split=sentence_split)
    per_head_pairwise = {}
    done = 0

    for l in range(n_layers):
        for h in range(n_heads):
            done += 1
            if done % 24 == 0:
                print(f"  Pairwise probing head {done}/{total}", flush=True)

            pw_data = pw_dataset[(l, h)]
            X_pairs = pw_data["X_pairs"]

            if X_pairs.shape[0] < 20:
                per_head_pairwise[(l, h)] = {
                    "arc": None, "relation": None,
                }
                continue

            arc_result = pw_prober.probe_arc(X_pairs, pw_data["y_arc"], pw_data["sentence_id"])
            rel_result = pw_prober.probe_relation(
                pw_data["X_pairs_pos"], pw_data["y_rel_pos"], pw_data["sentence_id_pos"],
            )

            per_head_pairwise[(l, h)] = {
                "arc": arc_result,
                "relation": rel_result,
            }

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
    print(f"\n{'='*60}", flush=True)
    print(f"  Probing Results: {model_label}", flush=True)
    print(f"{'='*60}", flush=True)

    # Word-embedding baseline
    emb = results["word_embedding_baseline"]
    print(f"\n  Word-Embedding Baseline:", flush=True)
    if emb["pos"]:
        print(f"    POS accuracy:     {emb['pos']['accuracy']:.3f}", flush=True)
    if emb["dep_rel"]:
        print(f"    Dep-rel accuracy: {emb['dep_rel']['accuracy']:.3f}", flush=True)
    if emb["depth"]:
        print(f"    Depth spearman r: {emb['depth']['spearman_r']:.3f}", flush=True)

    # Layer-wise summary
    print(f"\n  Layer-wise mean accuracy / spearman_r:", flush=True)
    layer_summ = results["layer_summary"]
    for l in range(len(layer_summ["pos"])):
        pos_v = layer_summ["pos"][l]
        dep_v = layer_summ["dep_rel"][l]
        depth_v = layer_summ["depth"][l]
        print(f"    Layer {l:2d}:  POS={pos_v:.3f}  Dep={dep_v:.3f}  Depth={depth_v:.3f}", flush=True)

    # Top heads per property (token-level)
    for prop in ["pos", "dep_rel", "depth"]:
        metric = "accuracy" if prop != "depth" else "spearman_r"
        scored = []
        for (l, h), res in results["per_head"].items():
            if res[prop] is not None:
                scored.append((l, h, res[prop][metric]))
        scored.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Top-5 heads for {prop} ({metric}):", flush=True)
        for l, h, v in scored[:5]:
            print(f"    Layer {l}, Head {h}: {v:.3f}", flush=True)

    # Pairwise results
    pw = results.get("per_head_pairwise", {})
    if pw:
        pw_baselines = results.get("pairwise_baselines", {})
        print(f"\n  Pairwise Baselines:", flush=True)
        if pw_baselines.get("word_emb_arc"):
            print(f"    Word-emb arc F1:      {pw_baselines['word_emb_arc']['f1']:.3f}", flush=True)
        if pw_baselines.get("word_emb_rel"):
            print(f"    Word-emb rel acc:     {pw_baselines['word_emb_rel']['accuracy']:.3f}", flush=True)
        if pw_baselines.get("distance_arc"):
            print(f"    Distance arc F1:      {pw_baselines['distance_arc']['f1']:.3f}", flush=True)

        pw_summary = results.get("layer_summary_pairwise", {})
        if pw_summary:
            print(f"\n  Layer-wise pairwise (arc F1 / rel acc):", flush=True)
            n = len(pw_summary.get("arc", []))
            for l in range(n):
                arc_v = pw_summary["arc"][l]
                rel_v = pw_summary["relation"][l]
                print(f"    Layer {l:2d}:  Arc={arc_v:.3f}  Rel={rel_v:.3f}", flush=True)

        for task, metric in [("arc", "f1"), ("relation", "accuracy")]:
            scored = []
            for (l, h), res in pw.items():
                if res[task] is not None and metric in res[task]:
                    scored.append((l, h, res[task][metric]))
            scored.sort(key=lambda x: x[2], reverse=True)
            print(f"\n  Top-5 heads for pairwise {task} ({metric}):", flush=True)
            for l, h, v in scored[:5]:
                print(f"    Layer {l}, Head {h}: {v:.3f}", flush=True)


def print_divergence_results(divergence, correlations):
    print(f"\n{'='*60}", flush=True)
    print(f"  Probing Divergence (Translator - Impossible)", flush=True)
    print(f"{'='*60}", flush=True)
    metric_names = {
        "pos": "ΔAccuracy", "dep_rel": "ΔAccuracy", "depth": "ΔSpearman_r",
        "pairwise_arc": "ΔF1", "pairwise_relation": "ΔAccuracy",
    }
    for prop, delta in divergence.items():
        metric = metric_names.get(prop, "Δmetric")
        print(f"\n  {prop} ({metric}):", flush=True)
        print(f"    Mean:  {np.mean(delta):.4f}", flush=True)
        print(f"    Max:   {np.max(delta):.4f}", flush=True)
        print(f"    Min:   {np.min(delta):.4f}", flush=True)

        # Top divergent heads
        flat = [(l, h, delta[l, h]) for l in range(delta.shape[0]) for h in range(delta.shape[1])]
        flat.sort(key=lambda x: x[2], reverse=True)
        print(f"    Top-5 positive divergence heads:", flush=True)
        for l, h, v in flat[:5]:
            print(f"      Layer {l}, Head {h}: {v:+.4f}", flush=True)

    print(f"\n  Entropy-Probing Correlation (Eq. 7):", flush=True)
    for prop, c in correlations.items():
        if c["rho"] is None or c["p_value"] is None:
            print(f"    {prop}: not available", flush=True)
            continue
        sig = "***" if c["significant"] else "n.s."
        print(f"    {prop}: rho={c['rho']:.3f}, p={c['p_value']:.2e} {sig}", flush=True)


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
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--spacy_model", type=str, default="en_core_web_sm",
        help="spaCy model for syntactic parsing (default: en_core_web_sm)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for GPU forward passes",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Load models with fp16 weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = detect_best_device() if args.device == "auto" else torch.device(args.device)
    tokenizer_path = args.tokenizer or args.impossible_model

    # Load dataset
    scrambled_sentences, original_sentences = load_dataset(
        args.dataset, args.max_sentences
    )
    print(f"Loaded {len(scrambled_sentences)} sentence pairs", flush=True)
    sentence_split = build_sentence_split(len(scrambled_sentences))

    # Syntactic labeler
    labeler = SyntacticLabeler(args.spacy_model)
    syntax_cache = prepare_syntax_cache(
        labeler,
        scrambled_sentences,
        original_sentences,
        batch_size=max(64, args.batch_size * 4),
    )

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
        batch_size=args.batch_size,
        fp16=args.fp16,
        syntax_cache=syntax_cache,
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
        batch_size=args.batch_size,
        fp16=args.fp16,
        syntax_cache=syntax_cache,
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
        batch_size=args.batch_size,
        fp16=args.fp16,
        syntax_cache=syntax_cache,
    )
    print_probing_results(results_base, "GPT-2 Base")

    # ---- Divergence analysis ----
    # Determine grid size from first model
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
        print("\n  Loading entropy results for correlation analysis...", flush=True)
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
        "sentence_split": {k: v.tolist() for k, v in sentence_split.items()},
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

    print(f"\nResults saved to {args.output}", flush=True)


# python prob_classifier.py \
#   --translation_model models/gutenberg-localShuffle-w3 \
#   --impossible_model mission-impossible-lms/local-shuffle-w3-gpt2 \
#   --base_model gpt2 \
#   --dataset test_data/training_data_1k_gutenberg_localShuffle.json \
#   --entropy_results entropy_impossible_results.json \
#   --output probing_results.json \
#   --max_sentences 200

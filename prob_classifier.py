import argparse
import json
import time
from collections import defaultdict, deque

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Device detection and global perf knobs
# =============================================================================

def detect_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            if mps.is_available() and mps.is_built():
                return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


# =============================================================================
# Syntactic labelling (batched spaCy)
# =============================================================================

class SyntacticLabeler:
    """Batched spaCy parsing for gold labels, and batched tokenization for
    scrambled surface-form segmentation."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    @staticmethod
    def _tree_depth(tok) -> int:
        d, cur = 0, tok
        while cur.head != cur:
            d += 1
            cur = cur.head
        return d

    def _doc_to_labels(self, doc):
        toks, pos, dep, head, depth = [], [], [], [], []
        for t in doc:
            toks.append(t.text)
            pos.append(t.pos_)
            dep.append(t.dep_)
            head.append(t.head.i)
            depth.append(self._tree_depth(t))
        return {
            "tokens": toks,
            "pos": pos,
            "dep_rel": dep,
            "head_idx": head,
            "depth": depth,
        }

    def parse_batch(self, sentences, batch_size: int = 256):
        out = []
        for doc in tqdm(
            self.nlp.pipe(sentences, batch_size=batch_size),
            total=len(sentences), desc="spaCy parse", unit="sent",
        ):
            out.append(self._doc_to_labels(doc))
        return out

    def tokenize_batch(self, sentences, batch_size: int = 512):
        return list(tqdm(
            self.nlp.tokenizer.pipe(sentences, batch_size=batch_size),
            total=len(sentences), desc="spaCy tokenize", unit="sent",
        ))


# =============================================================================
# Identity-based subword alignment (probes.tex §3.2)
# =============================================================================

def _first_subword_covering(offsets, ch_start: int, ch_end: int):
    for idx, (s, e) in enumerate(offsets):
        if e <= ch_start:
            continue
        if s >= ch_end:
            break
        if max(s, ch_start) < min(e, ch_end):
            return idx
    return None


def align_identity(scrambled_doc, offsets, gold_labels):
    """FIFO identity alignment. See probes.tex §3.2."""
    pool = defaultdict(deque)
    for idx, text in enumerate(gold_labels["tokens"]):
        pool[text].append(idx)

    aligned = []
    for sw_pos, tok in enumerate(scrambled_doc):
        dq = pool.get(tok.text)
        if not dq:
            continue
        orig_idx = dq.popleft()
        sw_idx = _first_subword_covering(offsets, tok.idx, tok.idx + len(tok))
        if sw_idx is None:
            continue
        aligned.append({
            "subword_idx": sw_idx,
            "scrambled_word_idx": sw_pos,
            "word_idx": orig_idx,
            "pos": gold_labels["pos"][orig_idx],
            "dep_rel": gold_labels["dep_rel"][orig_idx],
            "head_idx": gold_labels["head_idx"][orig_idx],
            "depth": gold_labels["depth"][orig_idx],
        })
    return aligned


# =============================================================================
# Batched GPU head-representation extractor
# =============================================================================

class HeadRepresentationExtractor:
    """Captures per-head outputs BEFORE c_proj via a forward pre-hook and
    runs batched forward passes on GPU. Returns representations as a single
    tensor on the target device; fp16 on CUDA to halve memory & speed up
    matmul-heavy kernels."""

    def __init__(self, model_path: str, tokenizer_path: str = None,
                 device: torch.device = None, fp16: bool = True):
        self.device = device or detect_best_device()

        tok_path = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        want_fp16 = fp16 and self.device.type == "cuda"
        dtype = torch.float16 if want_fp16 else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                output_hidden_states=False,
                torch_dtype=dtype,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="eager",
                output_hidden_states=False,
                dtype=dtype,
            )
        self.model.to(self.device).eval()

        cfg = self.model.config
        self.n_layers = cfg.n_layer
        self.n_heads = cfg.n_head
        self.n_embd = cfg.n_embd
        self.d_head = cfg.n_embd // cfg.n_head
        self.fp16 = want_fp16

        self._captured = {}
        self._hooks = []
        for l in range(self.n_layers):
            c_proj = self.model.transformer.h[l].attn.c_proj
            self._hooks.append(
                c_proj.register_forward_pre_hook(self._make_prehook(l))
            )

    def _make_prehook(self, layer_idx: int):
        def fn(module, inputs):
            pre = inputs[0]  # (B, S, n_embd) BEFORE c_proj
            B, S, _ = pre.shape
            per_head = pre.view(B, S, self.n_heads, self.d_head)
            self._captured[layer_idx] = per_head.detach()
        return fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ---- word embedding access (for baselines) ----
    @torch.inference_mode()
    def word_embeddings(self, input_ids):
        return self.model.transformer.wte(input_ids)

    # ---- tokenization helper ----
    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

    # ---- batched forward that returns per-head outputs on GPU ----
    @torch.inference_mode()
    def forward_batch(self, texts):
        """Runs one padded forward pass. Returns:
            per_head: (L, B, S, H, d_head) fp32 on self.device
            attention_mask: (B, S) int
            offsets: list of list of (start,end) char spans per sentence
            token_ids: (B, S) int on self.device
        """
        self._captured.clear()
        enc = self.tokenize(texts)
        offset_mapping = enc.pop("offset_mapping")  # CPU tensor
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        self.model(**inputs)

        # Stack captured tensors along a new leading "layer" dim, cast to fp32
        # for downstream probes. Shape: (L, B, S, H, d_head).
        stacked = torch.stack(
            [self._captured[l] for l in range(self.n_layers)], dim=0
        ).float()
        self._captured.clear()

        offsets_per_sent = []
        for b in range(offset_mapping.shape[0]):
            row = offset_mapping[b].tolist()
            offsets_per_sent.append([tuple(x) for x in row])

        return (
            stacked,
            inputs["attention_mask"],
            offsets_per_sent,
            inputs["input_ids"],
        )


# =============================================================================
# Global dataset builder: one pass over the corpus, returns GPU tensors
# =============================================================================

class ProbingTensors:
    """Container for the token-level probing tensors of a single model.

    Attributes (all on extractor.device unless noted):
        reps      : (L, H, N, d_head) fp32  per-head outputs at aligned tokens
        wte       : (N, n_embd)       fp32  word-embedding feature for baseline
        y_pos     : (N,) int64
        y_dep     : (N,) int64
        y_depth   : (N,) float32
        sent_id   : (N,) int64
        word_idx  : (N,) int64  (original-parse index, used for pairwise arcs)
        head_word : (N,) int64  (original-parse head index of this token)
    """
    def __init__(self, reps, wte, y_pos, y_dep, y_depth,
                 sent_id, word_idx, head_word):
        self.reps = reps
        self.wte = wte
        self.y_pos = y_pos
        self.y_dep = y_dep
        self.y_depth = y_depth
        self.sent_id = sent_id
        self.word_idx = word_idx
        self.head_word = head_word


def build_probing_tensors(
    extractor: HeadRepresentationExtractor,
    scrambled_sentences,
    scrambled_docs,
    original_labels,
    pos_enc: LabelEncoder,
    dep_enc: LabelEncoder,
    batch_size: int = 32,
) -> ProbingTensors:
    device = extractor.device
    L, H, dh = extractor.n_layers, extractor.n_heads, extractor.d_head

    reps_chunks = []          # list of (L, n_aligned_in_batch, H, dh) tensors
    wte_chunks = []
    y_pos_list, y_dep_list, y_depth_list = [], [], []
    sent_list, word_idx_list, head_word_list = [], [], []

    n_total = len(scrambled_sentences)
    unk_pos = len(pos_enc.classes_)  # reserved for unseen (shouldn't happen)
    unk_dep = len(dep_enc.classes_)

    for start in tqdm(range(0, n_total, batch_size),
                      desc="extract", unit="batch"):
        end = min(start + batch_size, n_total)
        batch_texts = scrambled_sentences[start:end]

        per_head, attn_mask, offsets_list, input_ids = \
            extractor.forward_batch(batch_texts)
        # per_head shape: (L, B, S, H, dh)
        wte_full = extractor.word_embeddings(input_ids).float()  # (B, S, E)

        B = per_head.shape[1]
        # Gather aligned-token indices per sentence
        for b in range(B):
            i = start + b
            aligned = align_identity(
                scrambled_docs[i], offsets_list[b], original_labels[i]
            )
            if not aligned:
                continue
            sw_idxs = [a["subword_idx"] for a in aligned
                       if a["subword_idx"] < int(attn_mask[b].sum())]
            kept = [a for a in aligned
                    if a["subword_idx"] < int(attn_mask[b].sum())]
            if not kept:
                continue
            idx_tensor = torch.tensor(sw_idxs, device=device, dtype=torch.long)

            # per_head[:, b] is (L, S, H, dh); select aligned subword positions along S.
            sel = per_head[:, b, :, :, :].index_select(1, idx_tensor)  # (L, K, H, dh)
            reps_chunks.append(sel)

            # Word-embedding features for baseline (same subwords)
            wte_chunks.append(wte_full[b].index_select(0, idx_tensor))

            for a in kept:
                pos_id = pos_enc.transform([a["pos"]])[0] if a["pos"] in pos_enc.classes_ else unk_pos
                dep_id = dep_enc.transform([a["dep_rel"]])[0] if a["dep_rel"] in dep_enc.classes_ else unk_dep
                y_pos_list.append(int(pos_id))
                y_dep_list.append(int(dep_id))
                y_depth_list.append(float(a["depth"]))
                sent_list.append(i)
                word_idx_list.append(a["word_idx"])
                head_word_list.append(a["head_idx"])

    if not reps_chunks:
        raise RuntimeError("No aligned tokens produced — alignment failed on every sentence.")

    # reps_chunks[k] has shape (L, K_k, H, dh); concatenate along K:
    reps = torch.cat(reps_chunks, dim=1)            # (L, N, H, dh)
    reps = reps.permute(0, 2, 1, 3).contiguous()    # (L, H, N, dh)
    wte = torch.cat(wte_chunks, dim=0).contiguous() # (N, n_embd)

    y_pos = torch.tensor(y_pos_list, device=device, dtype=torch.long)
    y_dep = torch.tensor(y_dep_list, device=device, dtype=torch.long)
    y_depth = torch.tensor(y_depth_list, device=device, dtype=torch.float32)
    sent_id = torch.tensor(sent_list, device=device, dtype=torch.long)
    word_idx = torch.tensor(word_idx_list, device=device, dtype=torch.long)
    head_word = torch.tensor(head_word_list, device=device, dtype=torch.long)

    return ProbingTensors(reps, wte, y_pos, y_dep, y_depth,
                          sent_id, word_idx, head_word)


# =============================================================================
# GPU-batched probe trainers
# =============================================================================

def _standardize(X_all, train_mask):
    """Z-score per head using stats on the training partition.
    X_all: (LH, N, d) — returns same shape, plus (mean, std) on train."""
    Xt = X_all[:, train_mask, :]
    mean = Xt.mean(dim=1, keepdim=True)
    std = Xt.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (X_all - mean) / std


@torch.no_grad()
def _eval_clf(W, b, X_test, y_test):
    # W: (LH, d, C), b: (LH, C), X_test: (LH, Nte, d), y_test: (Nte,)
    logits = torch.bmm(X_test, W) + b.unsqueeze(1)
    pred = logits.argmax(dim=-1)                 # (LH, Nte)
    acc = (pred == y_test.unsqueeze(0)).float().mean(dim=1)  # (LH,)
    return pred, acc


def train_clf_probes_batched(X_all, y, train_mask, test_mask, n_classes,
                              epochs: int = 12, lr: float = 1e-2,
                              weight_decay: float = 1e-4,
                              minibatch: int = 4096,
                              val_frac: float = 0.1, patience: int = 2,
                              seed: int = 0):
    """Train LH linear classifiers in parallel with AdamW, per-head early
    stopping on a held-out validation slice of the training partition.

    Matches the probes.tex recipe: AdamW, lr=1e-2, <=12 epochs, mini-batches
    up to 4096, validation split + early stopping (patience=2, best-on-val
    weights restored per head).

    X_all: (LH, N, d) fp32 on device
    y:     (N,)       long
    """
    device = X_all.device
    LH, N, d = X_all.shape

    X_all = _standardize(X_all, train_mask)

    # Sentence-level split was already applied to build train_mask/test_mask.
    # Per probes.tex §3.4: carve an i.i.d. val slice only if the train partition
    # has at least 100 examples; otherwise monitor fit loss directly.
    g = torch.Generator(device=device).manual_seed(seed)
    tr_idx_all = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    use_earlystop = tr_idx_all.numel() >= 100
    perm_all = tr_idx_all[torch.randperm(tr_idx_all.numel(), generator=g, device=device)]
    if use_earlystop:
        n_val = max(1, int(val_frac * perm_all.numel()))
        val_idx = perm_all[:n_val]
        tr_idx = perm_all[n_val:]
    else:
        val_idx = perm_all[:0]
        tr_idx = perm_all

    X_tr = X_all.index_select(1, tr_idx).contiguous()   # (LH, Ntr, d)
    X_te = X_all[:, test_mask, :].contiguous()          # (LH, Nte, d)
    y_tr = y[tr_idx]
    y_te = y[test_mask]
    Ntr = y_tr.shape[0]
    if use_earlystop:
        X_val = X_all.index_select(1, val_idx).contiguous()
        y_val = y[val_idx]
        Nval = y_val.shape[0]

    W = torch.zeros(LH, d, n_classes, device=device, requires_grad=True)
    b = torch.zeros(LH, n_classes, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    best_val = torch.full((LH,), float("inf"), device=device)
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    bad = torch.zeros(LH, dtype=torch.long, device=device)

    for _ in range(epochs):
        perm = torch.randperm(Ntr, device=device)
        for s in range(0, Ntr, minibatch):
            idx = perm[s:s + minibatch]
            xb = X_tr[:, idx, :]                                # (LH, B, d)
            yb = y_tr[idx]                                      # (B,)
            logits = torch.bmm(xb, W) + b.unsqueeze(1)          # (LH, B, C)
            loss = F.cross_entropy(
                logits.reshape(-1, n_classes),
                yb.repeat(LH),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            if use_earlystop:
                val_logits = torch.bmm(X_val, W) + b.unsqueeze(1)   # (LH, Nval, C)
                val_loss_ph = F.cross_entropy(
                    val_logits.reshape(-1, n_classes),
                    y_val.repeat(LH),
                    reduction="none",
                ).reshape(LH, Nval).mean(dim=-1)                    # (LH,)
            else:
                fit_logits = torch.bmm(X_tr, W) + b.unsqueeze(1)
                val_loss_ph = F.cross_entropy(
                    fit_logits.reshape(-1, n_classes),
                    y_tr.repeat(LH),
                    reduction="none",
                ).reshape(LH, Ntr).mean(dim=-1)

            improved = val_loss_ph < best_val
            best_val = torch.where(improved, val_loss_ph, best_val)
            best_W = torch.where(improved.view(LH, 1, 1), W.detach(), best_W)
            best_b = torch.where(improved.view(LH, 1), b.detach(), best_b)
            bad = torch.where(improved, torch.zeros_like(bad), bad + 1)
            if bool((bad >= patience).all().item()):
                break

    pred, acc = _eval_clf(best_W, best_b, X_te, y_te)
    f1 = _weighted_f1(pred, y_te, n_classes)
    return {"pred": pred, "acc": acc, "f1": f1,
            "y_train": y_tr, "y_test": y_te}


def _weighted_f1(pred, y_test, n_classes):
    """pred: (LH, Nte); y_test: (Nte,). Returns (LH,) weighted F1."""
    LH, Nte = pred.shape
    device = pred.device
    f1 = torch.zeros(LH, device=device)
    n_support_total = y_test.shape[0]
    for c in range(n_classes):
        y_c = (y_test == c)
        support = y_c.sum()
        if support == 0:
            continue
        p_c = (pred == c)
        tp = (p_c & y_c.unsqueeze(0)).sum(dim=1).float()
        fp = (p_c & ~y_c.unsqueeze(0)).sum(dim=1).float()
        fn = (~p_c & y_c.unsqueeze(0)).sum(dim=1).float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1_c = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
        f1 += f1_c * (support.float() / n_support_total)
    return f1


def linreg_probes_batched(X_all, y, train_mask, test_mask,
                           epochs: int = 12, lr: float = 1e-2,
                           weight_decay: float = 1e-4,
                           minibatch: int = 4096,
                           val_frac: float = 0.1, patience: int = 2,
                           seed: int = 0):
    """Train LH linear regressors in parallel with AdamW + early stopping.

    Matches probes.tex recipe (AdamW, lr=1e-2, <=12 epochs, minibatches
    up to 4096, validation split). Returns per-head (mse, spearman).
    """
    device = X_all.device
    X_all = _standardize(X_all, train_mask)

    g = torch.Generator(device=device).manual_seed(seed)
    tr_idx_all = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    use_earlystop = tr_idx_all.numel() >= 100
    perm_all = tr_idx_all[torch.randperm(tr_idx_all.numel(), generator=g, device=device)]
    if use_earlystop:
        n_val = max(1, int(val_frac * perm_all.numel()))
        val_idx = perm_all[:n_val]
        tr_idx = perm_all[n_val:]
    else:
        val_idx = perm_all[:0]
        tr_idx = perm_all

    X_tr = X_all.index_select(1, tr_idx).contiguous()   # (LH, Ntr, d)
    X_te = X_all[:, test_mask, :].contiguous()          # (LH, Nte, d)
    y_tr = y[tr_idx].to(X_tr.dtype)
    y_te = y[test_mask].to(X_tr.dtype)
    if use_earlystop:
        X_val = X_all.index_select(1, val_idx).contiguous()
        y_val = y[val_idx].to(X_tr.dtype)

    LH, Ntr, d = X_tr.shape
    Nval = y_val.shape[0] if use_earlystop else 0

    W = torch.zeros(LH, d, 1, device=device, requires_grad=True)
    b = torch.zeros(LH, 1, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    best_val = torch.full((LH,), float("inf"), device=device)
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    bad = torch.zeros(LH, dtype=torch.long, device=device)

    for _ in range(epochs):
        perm = torch.randperm(Ntr, device=device)
        for s in range(0, Ntr, minibatch):
            idx = perm[s:s + minibatch]
            xb = X_tr[:, idx, :]                           # (LH, B, d)
            yb = y_tr[idx]                                 # (B,)
            y_hat = (torch.bmm(xb, W).squeeze(-1) + b)     # (LH, B)
            loss = F.mse_loss(y_hat, yb.unsqueeze(0).expand_as(y_hat))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            if use_earlystop:
                y_hat_val = torch.bmm(X_val, W).squeeze(-1) + b      # (LH, Nval)
                val_mse_ph = ((y_hat_val - y_val.unsqueeze(0)) ** 2).mean(dim=-1)
            else:
                y_hat_fit = torch.bmm(X_tr, W).squeeze(-1) + b       # (LH, Ntr)
                val_mse_ph = ((y_hat_fit - y_tr.unsqueeze(0)) ** 2).mean(dim=-1)
            improved = val_mse_ph < best_val
            best_val = torch.where(improved, val_mse_ph, best_val)
            best_W = torch.where(improved.view(LH, 1, 1), W.detach(), best_W)
            best_b = torch.where(improved.view(LH, 1), b.detach(), best_b)
            bad = torch.where(improved, torch.zeros_like(bad), bad + 1)
            if bool((bad >= patience).all().item()):
                break

    with torch.no_grad():
        y_pred = torch.bmm(X_te, best_W).squeeze(-1) + best_b  # (LH, Nte)
    mse = ((y_pred - y_te.unsqueeze(0)) ** 2).mean(dim=1)
    rho = _spearman_batched(y_pred, y_te)
    return {"pred": y_pred, "mse": mse, "spearman": rho,
            "y_train": y_tr, "y_test": y_te}


def _rank_avg(x: torch.Tensor) -> torch.Tensor:
    """Exact average-rank (tie-aware) along last dim. Accepts (N,) or (..., N).

    Implements the same rank convention as scipy.stats.rankdata(method='average').
    """
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    B, N = x.shape
    order = x.argsort(dim=-1)
    x_sorted = x.gather(-1, order)
    # Group ids: start a new group wherever sorted value changes
    first = torch.ones(B, 1, dtype=torch.bool, device=x.device)
    eq = torch.cat(
        [first, x_sorted[..., 1:] != x_sorted[..., :-1]], dim=-1
    )
    group_id = eq.long().cumsum(dim=-1) - 1                     # (B, N)
    ord_rank = torch.arange(N, device=x.device, dtype=torch.float32).expand(B, N)
    sums = torch.zeros(B, N, device=x.device, dtype=torch.float32)
    sums.scatter_add_(-1, group_id, ord_rank)
    counts = torch.zeros(B, N, device=x.device, dtype=torch.float32)
    counts.scatter_add_(-1, group_id, torch.ones(B, N, device=x.device, dtype=torch.float32))
    avg_rank_by_group = sums / counts.clamp_min(1.0)
    sorted_ranks = avg_rank_by_group.gather(-1, group_id)       # ranks in sorted order
    ranks = torch.empty_like(sorted_ranks)
    ranks.scatter_(-1, order, sorted_ranks)
    if squeeze:
        ranks = ranks.squeeze(0)
    return ranks


def _spearman_batched(pred, target):
    """pred: (LH, N); target: (N,). Returns (LH,) spearman rho with
    exact average-rank tie handling (matches scipy rankdata('average'))."""
    pr = _rank_avg(pred)
    tr = _rank_avg(target).unsqueeze(0).expand_as(pr)
    pr = pr - pr.mean(dim=-1, keepdim=True)
    tr = tr - tr.mean(dim=-1, keepdim=True)
    num = (pr * tr).sum(dim=-1)
    denom = (pr.pow(2).sum(dim=-1).sqrt() * tr.pow(2).sum(dim=-1).sqrt())
    return num / denom.clamp_min(1e-9)


def _binary_metrics(pred, y_true):
    tp = ((pred == 1) & (y_true == 1)).sum().item()
    fp = ((pred == 1) & (y_true == 0)).sum().item()
    fn = ((pred == 0) & (y_true == 1)).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    acc = (pred == y_true).float().mean().item()
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _weighted_classification_metrics(pred, y_true, n_classes):
    acc = (pred == y_true).float().mean().item()
    f1_w = _weighted_f1(pred.unsqueeze(0), y_true, n_classes).item()
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
    }


def _majority_prediction(y_train, y_test):
    majority = torch.bincount(y_train, minlength=int(y_train.max().item()) + 1).argmax()
    return majority.repeat(y_test.shape[0])


def _random_predictions(y_test, n_classes, seed=0):
    g = torch.Generator(device=y_test.device).manual_seed(seed)
    return torch.randint(0, n_classes, (y_test.shape[0],), device=y_test.device, generator=g)


def _shuffle_subset_labels(y, mask, seed=0):
    out = y.clone()
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return out
    g = torch.Generator(device=y.device).manual_seed(seed)
    perm = idx[torch.randperm(idx.numel(), generator=g, device=y.device)]
    out[idx] = y[perm]
    return out


def _remap_observed_relation_labels(y_rel, pos_mask):
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
    if pos_idx.numel() == 0:
        return None, None, 0
    rel_classes, inverse = torch.unique(
        y_rel.index_select(0, pos_idx), sorted=True, return_inverse=True
    )
    remapped = torch.full_like(y_rel, -1)
    remapped[pos_idx] = inverse
    return remapped, rel_classes, int(rel_classes.numel())


# =============================================================================
# Baselines packaged in the expected result dict shape
# =============================================================================

def _classification_result_dict(acc, f1, n_classes, n_samples,
                                 majority_acc, random_label_acc):
    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "n_classes": int(n_classes),
        "n_samples": int(n_samples),
        "random_baseline": float(1.0 / max(1, n_classes)),
        "majority_baseline": float(majority_acc),
        "random_label_baseline": float(random_label_acc),
    }


def _depth_result_dict(mse, rho, n_samples, baseline_mse, random_label_mse):
    return {
        "mse": float(mse),
        "spearman_r": float(rho),
        "spearman_p": 0.0,
        "n_samples": int(n_samples),
        "mean_baseline_mse": float(baseline_mse),
        "random_label_mse": float(random_label_mse),
    }


# =============================================================================
# Pairwise probe — per-head, features built on GPU
# =============================================================================

def _build_pair_sentence_tensors(pt: ProbingTensors, max_pairs_per_sent=None):
    """Build per-token-pair label tensors once (arc labels + sentence_ids +
    token indices). Pair features (concat of reps) are built per-head to
    avoid materialising (LH, n_pairs, 2*d_head) at once.

    Returns:
        pair_i, pair_j : (P,) long  — indices into N (token-level)
        y_arc          : (P,) long  — 1 if head_of_i == word_idx_j in original parse
        y_rel          : (P,) long  — relation class id for positive pairs, else 0
        sent_id_pairs  : (P,) long
        pos_mask       : (P,) bool
        distance       : (P,) float — |scrambled_i - scrambled_j|  (proxy: subword distance)
    """
    device = pt.reps.device
    N = pt.sent_id.shape[0]
    sent_cpu = pt.sent_id.cpu().numpy()

    # Bucket token indices by sentence id
    by_sent = defaultdict(list)
    for tok_idx in range(N):
        by_sent[int(sent_cpu[tok_idx])].append(tok_idx)

    i_list, j_list = [], []
    for sid, idxs in by_sent.items():
        if len(idxs) < 2:
            continue
        arr = np.array(idxs)
        ii, jj = np.meshgrid(arr, arr, indexing='ij')
        mask = ii != jj
        i_list.append(ii[mask])
        j_list.append(jj[mask])

    if not i_list:
        return None
    pair_i = torch.tensor(np.concatenate(i_list), device=device, dtype=torch.long)
    pair_j = torch.tensor(np.concatenate(j_list), device=device, dtype=torch.long)

    head_of_i = pt.head_word[pair_i]       # (P,)
    word_idx_j = pt.word_idx[pair_j]
    y_arc = (head_of_i == word_idx_j).long()
    sent_id_pairs = pt.sent_id[pair_i]
    distance = (pair_i - pair_j).abs().float()

    # y_rel: dep-rel class of token i when arc, else sentinel (-1)
    y_rel = pt.y_dep[pair_i]
    pos_mask = (y_arc == 1)

    return {
        "pair_i": pair_i, "pair_j": pair_j, "y_arc": y_arc,
        "y_rel": y_rel, "sent_id": sent_id_pairs,
        "pos_mask": pos_mask, "distance": distance,
    }


def pairwise_features_for_head(reps_lh, pair_i, pair_j):
    """reps_lh: (N, d_head); returns (P, 2*d_head) concatenated pair features."""
    return torch.cat([reps_lh[pair_i], reps_lh[pair_j]], dim=-1)


def train_single_probe(X, y, train_mask, test_mask, n_classes,
                       epochs=12, lr=1e-2, weight_decay=1e-4,
                       minibatch=4096, class_weight=None,
                       val_frac: float = 0.1, patience: int = 2,
                       seed: int = 0):
    """Single-head classification probe on GPU (used for pairwise).

    Matches probes.tex recipe: AdamW, lr=1e-2, <=12 epochs, mini-batches
    up to 4096, val-split + early stopping (best-on-val weights restored).
    """
    device = X.device
    d = X.shape[1]

    Xt = X[train_mask]
    mean, std = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mean) / std

    # Per probes.tex §3.4: only hold out validation when train has >=100
    # examples; otherwise monitor fit loss directly.
    g = torch.Generator(device=device).manual_seed(seed)
    tr_idx_all = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    use_earlystop = tr_idx_all.numel() >= 100
    perm_all = tr_idx_all[torch.randperm(tr_idx_all.numel(), generator=g, device=device)]
    if use_earlystop:
        n_val = max(1, int(val_frac * perm_all.numel()))
        val_idx = perm_all[:n_val]
        tr_idx = perm_all[n_val:]
    else:
        val_idx = perm_all[:0]
        tr_idx = perm_all

    X_tr = Xn.index_select(0, tr_idx)
    X_te = Xn[test_mask]
    y_tr = y.index_select(0, tr_idx)
    y_te = y[test_mask]
    if use_earlystop:
        X_val = Xn.index_select(0, val_idx)
        y_val = y.index_select(0, val_idx)

    W = torch.zeros(d, n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    bad = 0

    Ntr = y_tr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(Ntr, device=device)
        for s in range(0, Ntr, minibatch):
            idx = perm[s:s + minibatch]
            xb, yb = X_tr[idx], y_tr[idx]
            logits = xb @ W + b
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        with torch.no_grad():
            if use_earlystop:
                val_logits = X_val @ W + b
                val_loss = F.cross_entropy(
                    val_logits, y_val, weight=class_weight
                ).item()
            else:
                fit_logits = X_tr @ W + b
                val_loss = F.cross_entropy(
                    fit_logits, y_tr, weight=class_weight
                ).item()
        if val_loss < best_val:
            best_val = val_loss
            best_W = W.detach().clone()
            best_b = b.detach().clone()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    with torch.no_grad():
        logits_te = X_te @ best_W + best_b
        pred = logits_te.argmax(-1)
    acc = (pred == y_te).float().mean().item()

    # f1 / precision / recall
    if n_classes == 2:
        tp = ((pred == 1) & (y_te == 1)).sum().item()
        fp = ((pred == 1) & (y_te == 0)).sum().item()
        fn = ((pred == 0) & (y_te == 1)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        return {"acc": acc, "precision": precision, "recall": recall,
                "f1": f1, "pred": pred, "y_test": y_te}
    else:
        f1_w = _weighted_f1(pred.unsqueeze(0), y_te, n_classes).item()
        return {"acc": acc, "f1_weighted": f1_w,
                "pred": pred, "y_test": y_te}


# =============================================================================
# Layer-wise aggregation and divergence
# =============================================================================

def layer_mean(values_LH, n_layers, n_heads):
    """values_LH: (L*H,) tensor or list."""
    arr = values_LH.reshape(n_layers, n_heads)
    return arr.mean(dim=1).tolist()


def compute_divergence(per_head_a, per_head_b, n_layers, n_heads):
    out = {}
    for prop in ("pos", "dep_rel", "depth"):
        d = np.zeros((n_layers, n_heads))
        for (l, h), ra in per_head_a.items():
            rb = per_head_b.get((l, h), {}).get(prop)
            if ra.get(prop) is None or rb is None:
                continue
            key = "spearman_r" if prop == "depth" else "accuracy"
            d[l, h] = ra[prop][key] - rb[key]
        out[prop] = d
    for task, metric in (("arc", "f1"), ("relation", "accuracy")):
        d = np.zeros((n_layers, n_heads))
        for (l, h), ra in per_head_a.items():
            ra_pw = ra.get("pairwise", {}).get(task)
            rb_pw = per_head_b.get((l, h), {}).get("pairwise", {}).get(task)
            if ra_pw is None or rb_pw is None:
                continue
            d[l, h] = ra_pw.get(metric, 0.0) - rb_pw.get(metric, 0.0)
        out[f"pairwise_{task}"] = d
    return out


def validate_entropy_correlation(entropy_div, probing_div):
    flat_e = np.asarray(entropy_div).flatten()
    out = {}
    for prop, d in probing_div.items():
        flat_d = d.flatten()
        if flat_e.shape != flat_d.shape:
            out[prop] = {"rho": None, "p_value": None, "significant": None}
            continue
        rho, p = stats.spearmanr(flat_e, flat_d)
        out[prop] = {
            "rho": float(rho), "p_value": float(p),
            "significant": bool(abs(rho) > 0.4 and p < 0.001),
        }
    return out


# =============================================================================
# Full pipeline per model
# =============================================================================

def run_probing_pipeline(model_path, tokenizer_path, scrambled_sentences,
                         original_labels, scrambled_docs,
                         pos_enc, dep_enc, sentence_split,
                         device, batch_size=32, fp16=True,
                         adam_epochs=12, pairwise_epochs=12,
                         label: str = "model"):

    print(f"\n{'='*60}\n  Probing: {label} ({model_path})\n{'='*60}", flush=True)

    extractor = HeadRepresentationExtractor(
        model_path, tokenizer_path=tokenizer_path,
        device=device, fp16=fp16,
    )
    L, H, dh = extractor.n_layers, extractor.n_heads, extractor.d_head

    t0 = time.time()
    pt = build_probing_tensors(
        extractor, scrambled_sentences, scrambled_docs, original_labels,
        pos_enc, dep_enc, batch_size=batch_size,
    )
    print(f"  Aligned tokens: {pt.y_pos.shape[0]}  "
          f"(extract {time.time()-t0:.1f}s)", flush=True)

    # ---- sentence-level train/test masks at the token level ----
    train_ids = torch.tensor(
        sentence_split["train"], device=device, dtype=torch.long,
    )
    test_ids = torch.tensor(
        sentence_split["test"], device=device, dtype=torch.long,
    )
    train_mask = torch.isin(pt.sent_id, train_ids)
    test_mask = torch.isin(pt.sent_id, test_ids)

    # ---- Token-level probes, all 144 heads in parallel ----
    reps_LHNd = pt.reps.reshape(L * H, -1, dh)   # (LH, N, d)

    n_pos = len(pos_enc.classes_)
    n_dep = len(dep_enc.classes_)

    print("  Training POS probes (batched, GPU)...", flush=True)
    t0 = time.time()
    pos_out = train_clf_probes_batched(
        reps_LHNd, pt.y_pos, train_mask, test_mask, n_pos,
        epochs=adam_epochs,
    )
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    print("  Training dep_rel probes (batched, GPU)...", flush=True)
    t0 = time.time()
    dep_out = train_clf_probes_batched(
        reps_LHNd, pt.y_dep, train_mask, test_mask, n_dep,
        epochs=adam_epochs,
    )
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    print("  Fitting depth probes (batched linear, GPU)...", flush=True)
    t0 = time.time()
    depth_out = linreg_probes_batched(
        reps_LHNd, pt.y_depth, train_mask, test_mask,
        epochs=adam_epochs,
    )
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    # ---- Random-label baselines (shuffle y on training partition) ----
    print("  Random-label control probes...", flush=True)
    y_pos_shuf = pt.y_pos.clone()
    y_dep_shuf = pt.y_dep.clone()
    y_depth_shuf = pt.y_depth.clone()
    perm = torch.randperm(pt.y_pos.shape[0], device=device)
    y_pos_shuf[train_mask] = pt.y_pos[train_mask][torch.randperm(train_mask.sum(), device=device)]
    y_dep_shuf[train_mask] = pt.y_dep[train_mask][torch.randperm(train_mask.sum(), device=device)]
    y_depth_shuf[train_mask] = pt.y_depth[train_mask][torch.randperm(train_mask.sum(), device=device)]

    rand_pos = train_clf_probes_batched(
        reps_LHNd, y_pos_shuf, train_mask, test_mask, n_pos,
        epochs=adam_epochs,
    )
    rand_dep = train_clf_probes_batched(
        reps_LHNd, y_dep_shuf, train_mask, test_mask, n_dep,
        epochs=adam_epochs,
    )
    rand_depth = linreg_probes_batched(
        reps_LHNd, y_depth_shuf, train_mask, test_mask,
        epochs=adam_epochs,
    )

    # ---- Majority + mean-depth baselines (analytic) ----
    y_tr_pos = pt.y_pos[train_mask]
    y_te_pos = pt.y_pos[test_mask]
    maj_pos_cls = torch.bincount(y_tr_pos, minlength=n_pos).argmax().item()
    maj_pos_acc = float((y_te_pos == maj_pos_cls).float().mean().item())

    y_tr_dep = pt.y_dep[train_mask]
    y_te_dep = pt.y_dep[test_mask]
    maj_dep_cls = torch.bincount(y_tr_dep, minlength=n_dep).argmax().item()
    maj_dep_acc = float((y_te_dep == maj_dep_cls).float().mean().item())

    y_tr_dep_depth = pt.y_depth[train_mask]
    mean_depth = y_tr_dep_depth.mean()
    mean_depth_mse = float(((pt.y_depth[test_mask] - mean_depth) ** 2).mean().item())

    # ---- Assemble per-head dict in the legacy schema ----
    per_head = {}
    for lh in range(L * H):
        l, h = divmod(lh, H)
        per_head[(l, h)] = {
            "pos": _classification_result_dict(
                pos_out["acc"][lh].item(), pos_out["f1"][lh].item(),
                n_pos, pt.y_pos.shape[0],
                maj_pos_acc, rand_pos["acc"][lh].item(),
            ),
            "dep_rel": _classification_result_dict(
                dep_out["acc"][lh].item(), dep_out["f1"][lh].item(),
                n_dep, pt.y_dep.shape[0],
                maj_dep_acc, rand_dep["acc"][lh].item(),
            ),
            "depth": _depth_result_dict(
                depth_out["mse"][lh].item(), depth_out["spearman"][lh].item(),
                pt.y_depth.shape[0], mean_depth_mse,
                rand_depth["mse"][lh].item(),
            ),
        }

    layer_summary = {
        "pos": layer_mean(pos_out["acc"], L, H),
        "dep_rel": layer_mean(dep_out["acc"], L, H),
        "depth": layer_mean(depth_out["spearman"], L, H),
    }

    # ---- Word-embedding baseline (token level) ----
    print("  Word-embedding baseline...", flush=True)
    wte = pt.wte.unsqueeze(0)   # (1, N, n_embd) -- single "head"
    wte_pos = train_clf_probes_batched(
        wte, pt.y_pos, train_mask, test_mask, n_pos, epochs=adam_epochs,
    )
    wte_dep = train_clf_probes_batched(
        wte, pt.y_dep, train_mask, test_mask, n_dep, epochs=adam_epochs,
    )
    wte_depth = linreg_probes_batched(
        wte, pt.y_depth, train_mask, test_mask,
        epochs=adam_epochs,
    )
    wte_rand_pos = train_clf_probes_batched(
        wte, y_pos_shuf, train_mask, test_mask, n_pos,
        epochs=adam_epochs,
    )
    wte_rand_dep = train_clf_probes_batched(
        wte, y_dep_shuf, train_mask, test_mask, n_dep,
        epochs=adam_epochs,
    )
    wte_rand_depth = linreg_probes_batched(
        wte, y_depth_shuf, train_mask, test_mask,
        epochs=adam_epochs,
    )

    emb_baseline = {
        "pos": _classification_result_dict(
            wte_pos["acc"][0].item(), wte_pos["f1"][0].item(),
            n_pos, pt.y_pos.shape[0],
            maj_pos_acc, wte_rand_pos["acc"][0].item(),
        ),
        "dep_rel": _classification_result_dict(
            wte_dep["acc"][0].item(), wte_dep["f1"][0].item(),
            n_dep, pt.y_dep.shape[0],
            maj_dep_acc, wte_rand_dep["acc"][0].item(),
        ),
        "depth": _depth_result_dict(
            wte_depth["mse"][0].item(), wte_depth["spearman"][0].item(),
            pt.y_depth.shape[0], mean_depth_mse,
            wte_rand_depth["mse"][0].item(),
        ),
    }

    # ---- Pairwise probing ----
    print("  Building pairwise pair index tensors...", flush=True)
    pair = _build_pair_sentence_tensors(pt)
    per_head_pairwise = {}
    pw_layer_arc = np.zeros(L * H)
    pw_layer_rel = np.zeros(L * H)

    if pair is not None:
        pair_i, pair_j = pair["pair_i"], pair["pair_j"]
        y_arc = pair["y_arc"]
        y_rel_raw = pair["y_rel"]
        sid = pair["sent_id"]
        pos_mask_arc = pair["pos_mask"]
        distance = pair["distance"]
        y_rel, rel_classes, n_rel = _remap_observed_relation_labels(
            y_rel_raw, pos_mask_arc
        )

        train_pair = torch.isin(sid, train_ids)
        test_pair = torch.isin(sid, test_ids)
        train_pair_pos = train_pair & pos_mask_arc
        test_pair_pos = test_pair & pos_mask_arc

        # Class-balanced weights for sparse arc (probes.tex §3.2)
        n_pos_arc = y_arc[train_pair].sum().item()
        n_neg_arc = train_pair.sum().item() - n_pos_arc
        if n_pos_arc > 0 and n_neg_arc > 0:
            cw = torch.tensor(
                [0.5 * (n_pos_arc + n_neg_arc) / max(1, n_neg_arc),
                 0.5 * (n_pos_arc + n_neg_arc) / max(1, n_pos_arc)],
                device=device, dtype=torch.float32,
            )
        else:
            cw = None

        # Baseline metrics for pairwise arc/relation.
        maj_arc_pred = _majority_prediction(y_arc[train_pair], y_arc[test_pair])
        maj_arc_metrics = _binary_metrics(maj_arc_pred, y_arc[test_pair])
        rand_arc_pred = _random_predictions(y_arc[test_pair], 2, seed=11)
        rand_arc_metrics = _binary_metrics(rand_arc_pred, y_arc[test_pair])
        y_arc_shuf = _shuffle_subset_labels(y_arc, train_pair, seed=17)

        maj_rel_metrics = None
        rand_rel_metrics = None
        y_rel_shuf = y_rel.clone() if y_rel is not None else None
        if y_rel is not None and train_pair_pos.sum().item() > 0 and test_pair_pos.sum().item() > 0:
            maj_rel_pred = _majority_prediction(
                y_rel[train_pair_pos], y_rel[test_pair_pos]
            )
            maj_rel_metrics = _weighted_classification_metrics(
                maj_rel_pred, y_rel[test_pair_pos], n_rel
            )
            maj_rel_metrics["n_classes"] = n_rel
            rand_rel_pred = _random_predictions(
                y_rel[test_pair_pos], n_rel, seed=19
            )
            rand_rel_metrics = _weighted_classification_metrics(
                rand_rel_pred, y_rel[test_pair_pos], n_rel
            )
            rand_rel_metrics["n_classes"] = n_rel
            y_rel_shuf = _shuffle_subset_labels(y_rel, train_pair_pos, seed=23)

        print(f"  Pairwise arc/relation probes ({L*H} heads)...", flush=True)
        for lh in tqdm(range(L * H), desc="pairwise", unit="head"):
            l, h = divmod(lh, H)
            reps_lh = pt.reps[l, h]              # (N, d_head)
            X_pair = pairwise_features_for_head(reps_lh, pair_i, pair_j)

            arc_res = train_single_probe(
                X_pair, y_arc, train_pair, test_pair,
                n_classes=2, epochs=pairwise_epochs,
                class_weight=cw,
            )
            arc_rand_label = train_single_probe(
                X_pair, y_arc_shuf, train_pair, test_pair,
                n_classes=2, epochs=pairwise_epochs,
                class_weight=cw, seed=lh + 101,
            )

            if (
                y_rel is not None and
                test_pair_pos.sum().item() >= 20 and
                train_pair_pos.sum().item() >= 20
            ):
                # Relation probe on positive pairs only
                rel_res = train_single_probe(
                    X_pair, y_rel, train_pair_pos, test_pair_pos,
                    n_classes=n_rel, epochs=pairwise_epochs,
                )
                rel_rand_label = train_single_probe(
                    X_pair, y_rel_shuf, train_pair_pos, test_pair_pos,
                    n_classes=n_rel, epochs=pairwise_epochs,
                    seed=lh + 313,
                )
                rel_dict = {
                    "accuracy": float(rel_res["acc"]),
                    "f1_weighted": float(rel_res.get("f1_weighted", 0.0)),
                    "n_classes": n_rel,
                    "n_samples": int(train_pair_pos.sum().item()
                                     + test_pair_pos.sum().item()),
                    "random_baseline": None if rand_rel_metrics is None
                    else float(rand_rel_metrics["accuracy"]),
                    "majority_baseline": None if maj_rel_metrics is None
                    else float(maj_rel_metrics["accuracy"]),
                    "random_label_baseline": float(rel_rand_label["acc"]),
                }
            else:
                rel_dict = None

            arc_dict = {
                "accuracy": float(arc_res["acc"]),
                "precision": float(arc_res["precision"]),
                "recall": float(arc_res["recall"]),
                "f1": float(arc_res["f1"]),
                "n_pairs": int(y_arc.shape[0]),
                "n_positive": int(y_arc.sum().item()),
                "positive_rate": float(y_arc.float().mean().item()),
                "random_baseline": float(rand_arc_metrics["f1"]),
                "majority_baseline": float(maj_arc_metrics["f1"]),
                "random_label_baseline": float(arc_rand_label["f1"]),
            }

            per_head_pairwise[(l, h)] = {"arc": arc_dict, "relation": rel_dict}
            per_head[(l, h)]["pairwise"] = {"arc": arc_dict, "relation": rel_dict}
            pw_layer_arc[lh] = arc_dict["f1"]
            pw_layer_rel[lh] = rel_dict["accuracy"] if rel_dict else 0.0

        # Pairwise word-embedding and distance baselines
        print("  Pairwise baselines (word-emb + distance)...", flush=True)
        X_pair_wte = pairwise_features_for_head(pt.wte, pair_i, pair_j)
        pw_wte_arc = train_single_probe(
            X_pair_wte, y_arc, train_pair, test_pair,
            n_classes=2, epochs=pairwise_epochs, class_weight=cw,
        )
        pw_wte_rel = None
        pw_rand_arc = train_single_probe(
            X_pair_wte, y_arc_shuf, train_pair, test_pair,
            n_classes=2, epochs=pairwise_epochs, class_weight=cw, seed=401,
        )
        pw_rand_rel = None
        if (
            y_rel is not None and
            test_pair_pos.sum().item() >= 20 and
            train_pair_pos.sum().item() >= 20
        ):
            pw_wte_rel = train_single_probe(
                X_pair_wte, y_rel, train_pair_pos, test_pair_pos,
                n_classes=n_rel, epochs=pairwise_epochs,
            )
            pw_rand_rel = train_single_probe(
                X_pair_wte, y_rel_shuf, train_pair_pos, test_pair_pos,
                n_classes=n_rel, epochs=pairwise_epochs, seed=409,
            )
        dist_feat = distance.unsqueeze(-1)
        pw_dist = train_single_probe(
            dist_feat, y_arc, train_pair, test_pair,
            n_classes=2, epochs=pairwise_epochs, class_weight=cw,
        )
        pairwise_baselines = {
            "word_emb_arc": {
                "accuracy": float(pw_wte_arc["acc"]),
                "precision": float(pw_wte_arc["precision"]),
                "recall": float(pw_wte_arc["recall"]),
                "f1": float(pw_wte_arc["f1"]),
                "random_baseline": float(rand_arc_metrics["f1"]),
                "majority_baseline": float(maj_arc_metrics["f1"]),
                "random_label_baseline": float(pw_rand_arc["f1"]),
            },
            "word_emb_rel": None if pw_wte_rel is None else {
                "accuracy": float(pw_wte_rel["acc"]),
                "f1_weighted": float(pw_wte_rel.get("f1_weighted", 0.0)),
                "n_classes": n_rel,
                "random_baseline": None if rand_rel_metrics is None
                else float(rand_rel_metrics["accuracy"]),
                "majority_baseline": None if maj_rel_metrics is None
                else float(maj_rel_metrics["accuracy"]),
                "random_label_baseline": None if pw_rand_rel is None
                else float(pw_rand_rel["acc"]),
            },
            "distance_arc": {
                "accuracy": float(pw_dist["acc"]),
                "precision": float(pw_dist["precision"]),
                "recall": float(pw_dist["recall"]),
                "f1": float(pw_dist["f1"]),
                "random_baseline": float(rand_arc_metrics["f1"]),
                "majority_baseline": float(maj_arc_metrics["f1"]),
                "random_label_baseline": None,
            },
            "majority_arc": maj_arc_metrics,
            "random_arc": rand_arc_metrics,
            "majority_rel": maj_rel_metrics,
            "random_rel": rand_rel_metrics,
        }
    else:
        pairwise_baselines = {
            "word_emb_arc": None,
            "word_emb_rel": None,
            "distance_arc": None,
            "majority_arc": None,
            "random_arc": None,
            "majority_rel": None,
            "random_rel": None,
        }

    layer_summary_pairwise = {
        "arc": pw_layer_arc.reshape(L, H).mean(axis=1).tolist(),
        "relation": pw_layer_rel.reshape(L, H).mean(axis=1).tolist(),
    }

    # ---- Clean up ----
    extractor.remove_hooks()
    del extractor, pt, reps_LHNd
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "per_head": per_head,
        "per_head_pairwise": per_head_pairwise,
        "layer_summary": layer_summary,
        "layer_summary_pairwise": layer_summary_pairwise,
        "word_embedding_baseline": emb_baseline,
        "pairwise_baselines": pairwise_baselines,
    }


# =============================================================================
# Serialisation
# =============================================================================

def _serialise_results(results):
    out = {
        "layer_summary": results["layer_summary"],
        "word_embedding_baseline": results["word_embedding_baseline"],
        "per_head": {},
    }
    for (l, h), v in results["per_head"].items():
        out["per_head"][f"L{l}_H{h}"] = v
    out["per_head_pairwise"] = {
        f"L{l}_H{h}": v for (l, h), v in results["per_head_pairwise"].items()
    }
    out["layer_summary_pairwise"] = results.get("layer_summary_pairwise", {})
    out["pairwise_baselines"] = results.get("pairwise_baselines", {})
    return out


# =============================================================================
# Label-encoder fitting (shared across all models)
# =============================================================================

def fit_global_encoders(original_labels):
    pos_tags, dep_tags = set(), set()
    for lab in original_labels:
        pos_tags.update(lab["pos"])
        dep_tags.update(lab["dep_rel"])
    pos_enc = LabelEncoder().fit(sorted(pos_tags))
    dep_enc = LabelEncoder().fit(sorted(dep_tags))
    return pos_enc, dep_enc


def build_sentence_split(n, test_size=0.2, random_state=42):
    ids = np.arange(n)
    if n < 3:
        return {"train": ids, "test": np.array([], dtype=int)}
    tr, te = train_test_split(
        ids, test_size=test_size, random_state=random_state, shuffle=True,
    )
    return {"train": np.asarray(tr), "test": np.asarray(te)}


# =============================================================================
# CLI
# =============================================================================

def load_dataset(path, max_sentences=None):
    with open(path) as f:
        pairs = json.load(f)
    if max_sentences:
        pairs = pairs[:max_sentences]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def parse_args():
    ap = argparse.ArgumentParser(
        description="GPU probing classifier for attention-head syntax"
    )
    ap.add_argument("--translation_model", required=True)
    ap.add_argument("--impossible_model", required=True)
    ap.add_argument("--base_model", default="gpt2")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--entropy_results", default=None)
    ap.add_argument("--output", default="probing_results.json")
    ap.add_argument("--max_sentences", type=int, default=None)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Forward-pass batch size")
    ap.add_argument("--no_fp16", action="store_true",
                    help="Disable fp16 even on CUDA")
    ap.add_argument("--adam_epochs", type=int, default=12,
                    help="Token-level probe max training epochs (AdamW, lr=1e-2, "
                         "val-split + early stopping)")
    ap.add_argument("--pairwise_epochs", type=int, default=12,
                    help="Pairwise probe max training epochs (AdamW, lr=1e-2, "
                         "val-split + early stopping)")
    return ap.parse_args()


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = detect_best_device() if args.device == "auto" \
        else torch.device(args.device)
    tokenizer_path = args.tokenizer or args.impossible_model
    fp16 = not args.no_fp16

    scrambled, originals = load_dataset(args.dataset, args.max_sentences)
    print(f"Loaded {len(scrambled)} sentence pairs", flush=True)

    # ---- spaCy parse & scrambled tokenization (batched, once) ----
    labeler = SyntacticLabeler(args.spacy_model)
    original_labels = labeler.parse_batch(originals)
    scrambled_docs = labeler.tokenize_batch(scrambled)

    pos_enc, dep_enc = fit_global_encoders(original_labels)
    print(f"  POS classes: {len(pos_enc.classes_)}  "
          f"dep_rel classes: {len(dep_enc.classes_)}", flush=True)

    sentence_split = build_sentence_split(len(scrambled))

    def run(label, path):
        return run_probing_pipeline(
            model_path=path,
            tokenizer_path=tokenizer_path,
            scrambled_sentences=scrambled,
            original_labels=original_labels,
            scrambled_docs=scrambled_docs,
            pos_enc=pos_enc, dep_enc=dep_enc,
            sentence_split=sentence_split,
            device=device,
            batch_size=args.batch_size,
            fp16=fp16,
            adam_epochs=args.adam_epochs,
            pairwise_epochs=args.pairwise_epochs,
            label=label,
        )

    results_t = run("Translator", args.translation_model)
    results_i = run("Impossible", args.impossible_model)
    results_b = run("GPT-2 Base", args.base_model)

    # ---- divergence ----
    L = max(l for (l, _) in results_t["per_head"]) + 1
    H = max(h for (_, h) in results_t["per_head"]) + 1
    div_t_i = compute_divergence(results_t["per_head"], results_i["per_head"], L, H)
    div_t_b = compute_divergence(results_t["per_head"], results_b["per_head"], L, H)
    div_b_i = compute_divergence(results_b["per_head"], results_i["per_head"], L, H)

    correlations = {
        k: {"rho": None, "p_value": None, "significant": None}
        for k in div_t_i
    }
    if args.entropy_results:
        with open(args.entropy_results) as f:
            ent = json.load(f)
        if "comparisons" in ent:
            delta_H = np.array(
                ent["comparisons"]["translation_vs_impossible"]["delta_H"]
            )
        else:
            delta_H = (np.array(ent["raw_entropy"]["H_impossible"])
                       - np.array(ent["raw_entropy"]["H_translation"]))
        correlations = validate_entropy_correlation(delta_H, div_t_i)

    output = {
        "models": {
            "translation": args.translation_model,
            "impossible": args.impossible_model,
            "base": args.base_model,
        },
        "tokenizer": tokenizer_path,
        "n_sentences": len(scrambled),
        "sentence_split": {k: v.tolist() for k, v in sentence_split.items()},
        "translator": _serialise_results(results_t),
        "impossible": _serialise_results(results_i),
        "base": _serialise_results(results_b),
        "divergence": {k: d.tolist() for k, d in div_t_i.items()},
        "divergence_vs_base": {k: d.tolist() for k, d in div_t_b.items()},
        "divergence_base_vs_impossible":
            {k: d.tolist() for k, d in div_b_i.items()},
        "entropy_probing_correlation": correlations,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}", flush=True)


if __name__ == "__main__":
    main()


# Example:
# python prob_classifier.py \
#   --translation_model models/gutenberg-localShuffle-w3 \
#   --impossible_model mission-impossible-lms/local-shuffle-w3-gpt2 \
#   --base_model gpt2 \
#   --dataset test_data/training_data_1k_gutenberg_localShuffle.json \
#   --entropy_results entropy_impossible_results.json \
#   --output probing_results.json \
#   --batch_size 64 --adam_epochs 12 --pairwise_epochs 12

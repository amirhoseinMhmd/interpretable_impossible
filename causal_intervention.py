"""
Causal intervention experiment (RQ3) for the impossible-language translator.

Implements layer-wise ablation studies on a fine-tuned translator GPT-2 to test
which layers are causally necessary for (a) translation quality and (b)
structural recovery, following the method specified in report/main.tex Section 3.

Three intervention types are supported:
    - zero    : replace the sublayer's output contribution with zeros
    - mean    : replace it with the dataset-mean activation
    - random  : replace it with Gaussian noise matched to activation statistics

Three intervention scopes are supported:
    - full    : ablate both attention and feed-forward sublayers
    - attn    : ablate only the attention sublayer
    - ffn     : ablate only the feed-forward sublayer

The script also runs progressive cumulative ablation (early / late / middle
windows), evaluates the intact and impossible-model baselines, and computes the
causal-vs-probing correlation against probing_results.json (if available).

Outputs everything to a single JSON file (default: causal_intervention_results.json).

Example
-------
python causal_intervention.py \
    --translator_model models/bnc_spoken-localShuffle-w3 \
    --impossible_model mission-impossible-lms/local-shuffle-w3-gpt2 \
    --base_model gpt2 \
    --dataset test_data/test_data_1k_bnc_spoken_wordHOP.json \
    --probing_results probing_results.json \
    --output causal_intervention_results.json \
    --num_sentences 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, GPT2LMHeadModel

try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

try:
    from scipy.stats import spearmanr
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InterventionConfig:
    """High level knobs for the experiment."""

    translator_model: str
    impossible_model: str
    base_model: str
    dataset: str
    output: str
    probing_results: Optional[str] = None
    num_sentences: int = 200
    max_new_tokens: int = 64
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    em_threshold: float = 0.05
    f1_threshold: float = 0.10
    bootstrap: int = 1000


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: str, n: int) -> List[Tuple[str, str]]:
    """Load (scrambled, reference) pairs from the project test_data JSON format."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    pairs: List[Tuple[str, str]] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            src, ref = str(entry[0]), str(entry[1])
            if src.strip() and ref.strip():
                pairs.append((src, ref))
    if n > 0:
        pairs = pairs[:n]
    return pairs


# ---------------------------------------------------------------------------
# Model wrapper with intervention hooks
# ---------------------------------------------------------------------------


SCOPE_FULL = "full"
SCOPE_ATTN = "attn"
SCOPE_FFN = "ffn"

KIND_ZERO = "zero"
KIND_MEAN = "mean"
KIND_RANDOM = "random"


class InterventionModel:
    """Wraps a GPT2LMHeadModel and exposes per-layer intervention hooks.

    The wrapper uses forward hooks on each transformer block's attention and
    MLP submodules. The original outputs are intercepted and replaced according
    to the active intervention specification. When no intervention is active
    the model behaves identically to the underlying HF model.

    Parameters
    ----------
    is_translator : bool
        If True, prompts are formatted as "Fix this text: {src}\nCorrected:"
        to match the fine-tuning format. Set False for the impossible-LM and
        base GPT-2 baselines which receive raw scrambled text.
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer,
        device: str,
        is_translator: bool = True,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.is_translator = is_translator
        self.n_layers = model.config.n_layer
        self.hidden_size = model.config.n_embd

        # Active intervention state. None means pass-through.
        # Separate dicts for attention and FFN hooks so SCOPE_FULL stores
        # correctly matched statistics for each sublayer independently.
        self._attn_interventions: Dict[int, dict] = {}
        self._ffn_interventions: Dict[int, dict] = {}

        # Cached statistics (per layer): mean and std of attn / ffn outputs
        self.attn_mean: Dict[int, torch.Tensor] = {}
        self.attn_std: Dict[int, torch.Tensor] = {}
        self.ffn_mean: Dict[int, torch.Tensor] = {}
        self.ffn_std: Dict[int, torch.Tensor] = {}

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._install_hooks()

    # ---- hook plumbing -------------------------------------------------

    def _install_hooks(self) -> None:
        for layer_idx, block in enumerate(self.model.transformer.h):
            attn = block.attn
            mlp = block.mlp

            def make_attn_hook(idx: int):
                def hook(module, inputs, output):
                    # GPT2Attention returns a tuple (attn_output, present, ...)
                    spec = self._attn_interventions.get(idx)
                    if spec is None:
                        return output
                    new_attn = self._apply_replacement(output[0], spec)
                    return (new_attn,) + output[1:]
                return hook

            def make_mlp_hook(idx: int):
                def hook(module, inputs, output):
                    spec = self._ffn_interventions.get(idx)
                    if spec is None:
                        return output
                    return self._apply_replacement(output, spec)
                return hook

            self._handles.append(attn.register_forward_hook(make_attn_hook(layer_idx)))
            self._handles.append(mlp.register_forward_hook(make_mlp_hook(layer_idx)))

    def _apply_replacement(self, tensor: torch.Tensor, spec: dict) -> torch.Tensor:
        """Replace a sublayer output according to the spec dict."""
        kind = spec["kind"]
        if kind == KIND_ZERO:
            return torch.zeros_like(tensor)
        if kind == KIND_MEAN:
            mean = spec["mean"].to(tensor.device, dtype=tensor.dtype)
            return mean.expand_as(tensor).contiguous()
        if kind == KIND_RANDOM:
            mean = spec["mean"].to(tensor.device, dtype=tensor.dtype)
            std = spec["std"].to(tensor.device, dtype=tensor.dtype)
            noise = torch.randn_like(tensor) * std + mean
            return noise
        raise ValueError(f"Unknown intervention kind: {kind}")

    # ---- intervention scheduling --------------------------------------

    def clear(self) -> None:
        self._attn_interventions.clear()
        self._ffn_interventions.clear()

    def set_intervention(
        self,
        layers: Sequence[int],
        kind: str,
        scope: str,
    ) -> None:
        """Configure interventions for a set of layers."""
        self.clear()
        for layer_idx in layers:
            if scope == SCOPE_FULL:
                # Store separate specs per sublayer so each hook gets
                # statistics that match its own activation distribution.
                self._attn_interventions[layer_idx] = self._build_spec(
                    layer_idx, kind, SCOPE_ATTN
                )
                self._ffn_interventions[layer_idx] = self._build_spec(
                    layer_idx, kind, SCOPE_FFN
                )
            elif scope == SCOPE_ATTN:
                self._attn_interventions[layer_idx] = self._build_spec(
                    layer_idx, kind, SCOPE_ATTN
                )
            elif scope == SCOPE_FFN:
                self._ffn_interventions[layer_idx] = self._build_spec(
                    layer_idx, kind, SCOPE_FFN
                )
            else:
                raise ValueError(f"Unknown scope: {scope}")

    def _build_spec(self, layer_idx: int, kind: str, scope: str) -> dict:
        """Build a replacement spec for a single sublayer."""
        if kind == KIND_ZERO:
            return {"kind": kind}
        # mean / random need cached statistics
        if scope == SCOPE_ATTN:
            mean = self.attn_mean[layer_idx]
            std = self.attn_std[layer_idx]
        else:  # SCOPE_FFN
            mean = self.ffn_mean[layer_idx]
            std = self.ffn_std[layer_idx]
        return {"kind": kind, "mean": mean, "std": std}

    # ---- statistics estimation ----------------------------------------

    @torch.no_grad()
    def estimate_activation_statistics(
        self, texts: Sequence[str], max_length: int = 128
    ) -> None:
        """Run a forward pass over `texts` and record per-layer mean/std.

        The texts are formatted using the same prompt template as generation
        so that the captured statistics reflect the real activation
        distribution the translator sees at inference time.
        """
        attn_sum = {i: torch.zeros(self.hidden_size) for i in range(self.n_layers)}
        attn_sq  = {i: torch.zeros(self.hidden_size) for i in range(self.n_layers)}
        ffn_sum  = {i: torch.zeros(self.hidden_size) for i in range(self.n_layers)}
        ffn_sq   = {i: torch.zeros(self.hidden_size) for i in range(self.n_layers)}
        counts   = {i: 0 for i in range(self.n_layers)}

        # Temporary capture hooks (separate from intervention hooks).
        capture_handles = []
        for layer_idx, block in enumerate(self.model.transformer.h):
            def make_attn_capture(idx: int):
                def hook(module, inputs, output):
                    t = output[0].detach().float().cpu()
                    flat = t.reshape(-1, t.shape[-1])
                    attn_sum[idx] += flat.sum(dim=0)
                    attn_sq[idx]  += (flat * flat).sum(dim=0)
                    counts[idx]   += flat.shape[0]
                    return output
                return hook

            def make_mlp_capture(idx: int):
                def hook(module, inputs, output):
                    t = output.detach().float().cpu()
                    flat = t.reshape(-1, t.shape[-1])
                    ffn_sum[idx] += flat.sum(dim=0)
                    ffn_sq[idx]  += (flat * flat).sum(dim=0)
                    return output
                return hook

            capture_handles.append(
                block.attn.register_forward_hook(make_attn_capture(layer_idx))
            )
            capture_handles.append(
                block.mlp.register_forward_hook(make_mlp_capture(layer_idx))
            )

        # Temporarily disable any active interventions during capture.
        saved_attn = self._attn_interventions
        saved_ffn  = self._ffn_interventions
        self._attn_interventions = {}
        self._ffn_interventions  = {}

        try:
            for text in texts:
                if not text or not text.strip():
                    continue
                # Use the same prompt format as generation.
                formatted = self._format_prompt(text)
                enc = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                if enc["input_ids"].numel() == 0:
                    continue
                self.model(**enc)
        finally:
            for h in capture_handles:
                h.remove()
            self._attn_interventions = saved_attn
            self._ffn_interventions  = saved_ffn

        for i in range(self.n_layers):
            n = max(counts[i], 1)

            mean_attn = attn_sum[i] / n
            var_attn  = (attn_sq[i] / n) - mean_attn * mean_attn
            self.attn_mean[i] = mean_attn
            self.attn_std[i]  = torch.sqrt(var_attn.clamp(min=1e-8))

            mean_ffn = ffn_sum[i] / n
            var_ffn  = (ffn_sq[i] / n) - mean_ffn * mean_ffn
            self.ffn_mean[i] = mean_ffn
            self.ffn_std[i]  = torch.sqrt(var_ffn.clamp(min=1e-8))

    # ---- prompt formatting --------------------------------------------

    def _format_prompt(self, src: str) -> str:
        """Return the correctly formatted prompt for this model type."""
        if self.is_translator:
            # Must match the training format used in train.py exactly:
            # "Fix this text: {corrupted}\nCorrected: {correct}<|endoftext|>"
            return f"Fix this text: {src}\nCorrected:"
        return src  # impossible-LM and base GPT-2 see raw scrambled text

    # ---- generation ---------------------------------------------------

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], max_new_tokens: int = 64) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            if not prompt or not prompt.strip():
                outputs.append("")
                continue

            formatted = self._format_prompt(prompt)
            enc = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            if enc["input_ids"].numel() == 0:
                outputs.append("")
                continue

            gen = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = gen[0, enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text.strip())
        return outputs


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip() == ref.strip())


def token_accuracy(pred: str, ref: str) -> float:
    pt = pred.strip().split()
    rt = ref.strip().split()
    if not rt:
        return 0.0
    n = min(len(pt), len(rt))
    if n == 0:
        return 0.0
    correct = sum(1 for i in range(n) if pt[i] == rt[i])
    return correct / len(rt)


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(pred: str, ref: str, max_n: int = 4) -> Dict[str, float]:
    """Sentence-level BLEU-1..BLEU-4 with brevity penalty (no smoothing)."""
    pt = pred.strip().split()
    rt = ref.strip().split()
    if not pt or not rt:
        return {f"bleu{n}": 0.0 for n in range(1, max_n + 1)}
    precisions = []
    for n in range(1, max_n + 1):
        p_ngrams = _ngrams(pt, n)
        r_ngrams = _ngrams(rt, n)
        if not p_ngrams:
            precisions.append(0.0)
            continue
        ref_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for g in r_ngrams:
            ref_counts[g] += 1
        match = 0
        used: Dict[Tuple[str, ...], int] = defaultdict(int)
        for g in p_ngrams:
            if used[g] < ref_counts.get(g, 0):
                match += 1
                used[g] += 1
        precisions.append(match / len(p_ngrams))
    bp = math.exp(1 - len(rt) / len(pt)) if len(pt) < len(rt) else 1.0
    out = {}
    for n in range(1, max_n + 1):
        ps = precisions[:n]
        if min(ps) == 0:
            out[f"bleu{n}"] = 0.0
        else:
            log_p = sum(math.log(p) for p in ps) / n
            out[f"bleu{n}"] = bp * math.exp(log_p)
    return out


# ---- dependency-based metrics via spaCy ----


class DependencyEvaluator:
    """Computes dependency F1, UAS, LAS using spaCy parses of pred vs ref."""

    def __init__(self):
        if not _SPACY_OK:
            self.nlp = None
            return
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None

    def parse(self, text: str):
        if self.nlp is None or not text.strip():
            return None
        return self.nlp(text)

    def score(self, pred: str, ref: str) -> Dict[str, float]:
        if self.nlp is None:
            return {"dep_f1": 0.0, "uas": 0.0, "las": 0.0}
        pred_doc = self.parse(pred)
        ref_doc  = self.parse(ref)
        if pred_doc is None or ref_doc is None or len(ref_doc) == 0:
            return {"dep_f1": 0.0, "uas": 0.0, "las": 0.0}

        def arcs(doc, labelled: bool):
            out = set()
            for tok in doc:
                head_text = tok.head.text.lower()
                if labelled:
                    out.add((tok.text.lower(), head_text, tok.dep_))
                else:
                    out.add((tok.text.lower(), head_text))
            return out

        p_unl = arcs(pred_doc, False)
        r_unl = arcs(ref_doc,  False)

        # Unlabeled F1 -> dep_f1
        tp   = len(p_unl & r_unl)
        prec = tp / len(p_unl) if p_unl else 0.0
        rec  = tp / len(r_unl) if r_unl else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Token-aligned UAS / LAS over the shorter length
        n = min(len(pred_doc), len(ref_doc))
        if n == 0:
            return {"dep_f1": f1, "uas": 0.0, "las": 0.0}
        uas_correct = 0
        las_correct = 0
        for i in range(n):
            p_tok = pred_doc[i]
            r_tok = ref_doc[i]
            if p_tok.head.i == r_tok.head.i:
                uas_correct += 1
                if p_tok.dep_ == r_tok.dep_:
                    las_correct += 1
        return {"dep_f1": f1, "uas": uas_correct / n, "las": las_correct / n}


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    em: float
    token_acc: float
    bleu1: float
    bleu2: float
    bleu3: float
    bleu4: float
    dep_f1: float
    uas: float
    las: float
    n: int
    per_sentence: List[Dict] = field(default_factory=list)

    def to_dict(self, include_per_sentence: bool = False) -> dict:
        d = {
            "em":        self.em,
            "token_acc": self.token_acc,
            "bleu1":     self.bleu1,
            "bleu2":     self.bleu2,
            "bleu3":     self.bleu3,
            "bleu4":     self.bleu4,
            "dep_f1":    self.dep_f1,
            "uas":       self.uas,
            "las":       self.las,
            "n":         self.n,
        }
        if include_per_sentence:
            d["per_sentence"] = self.per_sentence
        return d


def evaluate(
    wrapper: InterventionModel,
    pairs: Sequence[Tuple[str, str]],
    dep_eval: DependencyEvaluator,
    max_new_tokens: int,
) -> EvaluationResult:
    preds = wrapper.generate([p[0] for p in pairs], max_new_tokens=max_new_tokens)
    em_vals, tok_vals = [], []
    bleu_vals = {f"bleu{n}": [] for n in range(1, 5)}
    dep_vals  = {"dep_f1": [], "uas": [], "las": []}
    per = []
    for (src, ref), pred in zip(pairs, preds):
        em = exact_match(pred, ref)
        ta = token_accuracy(pred, ref)
        bl = bleu_score(pred, ref)
        dp = dep_eval.score(pred, ref)
        em_vals.append(em)
        tok_vals.append(ta)
        for k in bleu_vals:
            bleu_vals[k].append(bl[k])
        for k in dep_vals:
            dep_vals[k].append(dp[k])
        per.append({"src": src, "ref": ref, "pred": pred, "em": em, **bl, **dp})
    return EvaluationResult(
        em=float(np.mean(em_vals)),
        token_acc=float(np.mean(tok_vals)),
        bleu1=float(np.mean(bleu_vals["bleu1"])),
        bleu2=float(np.mean(bleu_vals["bleu2"])),
        bleu3=float(np.mean(bleu_vals["bleu3"])),
        bleu4=float(np.mean(bleu_vals["bleu4"])),
        dep_f1=float(np.mean(dep_vals["dep_f1"])),
        uas=float(np.mean(dep_vals["uas"])),
        las=float(np.mean(dep_vals["las"])),
        n=len(pairs),
        per_sentence=per,
    )


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values)
    rng = np.random.default_rng(0)
    means = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def load_model(path: str, device: str) -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """Load a GPT-2 model. Local fine-tuned checkpoints in this project ship
    without tokenizer files, so we always fall back to the standard ``gpt2``
    tokenizer (vocab_size=50257) when the local one is empty/broken."""
    try:
        tok = AutoTokenizer.from_pretrained(path)
        if len(tok) < 100:
            raise ValueError(f"degenerate tokenizer at {path} (vocab={len(tok)})")
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(path)
    return model, tok


def run_single_layer_ablations(
    wrapper: InterventionModel,
    pairs: Sequence[Tuple[str, str]],
    dep_eval: DependencyEvaluator,
    cfg: InterventionConfig,
    baseline: EvaluationResult,
) -> Dict[str, dict]:
    """Run zero/mean/random ablations across all 12 layers and three scopes."""
    results: Dict[str, dict] = {}
    for kind in (KIND_ZERO, KIND_MEAN, KIND_RANDOM):
        for scope in (SCOPE_FULL, SCOPE_ATTN, SCOPE_FFN):
            key = f"{kind}_{scope}"
            results[key] = {}
            for layer in range(wrapper.n_layers):
                wrapper.set_intervention([layer], kind=kind, scope=scope)
                res = evaluate(wrapper, pairs, dep_eval, cfg.max_new_tokens)
                delta_em  = baseline.em     - res.em
                delta_f1  = baseline.dep_f1 - res.dep_f1
                results[key][str(layer)] = {
                    **res.to_dict(),
                    "delta_em":    delta_em,
                    "delta_dep_f1": delta_f1,
                    "critical": int(
                        delta_em > cfg.em_threshold or delta_f1 > cfg.f1_threshold
                    ),
                }
                print(
                    f"[{key}] layer {layer:2d}  EM={res.em:.3f}  "
                    f"depF1={res.dep_f1:.3f}  ΔEM={delta_em:+.3f}  ΔF1={delta_f1:+.3f}"
                )
    wrapper.clear()
    return results


def run_cumulative_ablations(
    wrapper: InterventionModel,
    pairs: Sequence[Tuple[str, str]],
    dep_eval: DependencyEvaluator,
    cfg: InterventionConfig,
    baseline: EvaluationResult,
) -> Dict[str, list]:
    """Progressive early/late/middle window ablations using zero scope=full."""
    results: Dict[str, list] = {"early": [], "late": [], "middle": []}
    n = wrapper.n_layers

    # Early: ablate layers 0..k
    for k in range(n):
        layers = list(range(0, k + 1))
        wrapper.set_intervention(layers, kind=KIND_ZERO, scope=SCOPE_FULL)
        res = evaluate(wrapper, pairs, dep_eval, cfg.max_new_tokens)
        results["early"].append({
            "layers":       layers,
            **res.to_dict(),
            "delta_em":     baseline.em     - res.em,
            "delta_dep_f1": baseline.dep_f1 - res.dep_f1,
        })
        print(f"[early] layers 0-{k}  EM={res.em:.3f}  depF1={res.dep_f1:.3f}")

    # Late: ablate layers k..n-1
    for k in range(n):
        layers = list(range(k, n))
        wrapper.set_intervention(layers, kind=KIND_ZERO, scope=SCOPE_FULL)
        res = evaluate(wrapper, pairs, dep_eval, cfg.max_new_tokens)
        results["late"].append({
            "layers":       layers,
            **res.to_dict(),
            "delta_em":     baseline.em     - res.em,
            "delta_dep_f1": baseline.dep_f1 - res.dep_f1,
        })
        print(f"[late] layers {k}-{n-1}  EM={res.em:.3f}  depF1={res.dep_f1:.3f}")

    # Middle: sliding windows of width 2 and 3
    for start in range(n):
        for width in (2, 3):
            end = start + width - 1
            if end >= n:
                continue
            layers = list(range(start, end + 1))
            wrapper.set_intervention(layers, kind=KIND_ZERO, scope=SCOPE_FULL)
            res = evaluate(wrapper, pairs, dep_eval, cfg.max_new_tokens)
            results["middle"].append({
                "layers":       layers,
                "width":        width,
                **res.to_dict(),
                "delta_em":     baseline.em     - res.em,
                "delta_dep_f1": baseline.dep_f1 - res.dep_f1,
            })
            print(
                f"[middle] layers {start}-{end}  EM={res.em:.3f}  depF1={res.dep_f1:.3f}"
            )

    wrapper.clear()
    return results


def compute_interactions(single: Dict[str, dict]) -> Dict[str, dict]:
    """Pairwise additive prediction: ΔPerf(l) + ΔPerf(l+1).

    Actual pairwise drops come from cumulative-middle results; this helper
    exposes the additive baseline so callers can compute the interaction term.
    """
    base   = single["zero_full"]
    layers = sorted(int(k) for k in base.keys())
    out    = {}
    for i in range(len(layers) - 1):
        l, lp = layers[i], layers[i + 1]
        out[f"{l},{lp}"] = {
            "additive_delta_em":     base[str(l)]["delta_em"]     + base[str(lp)]["delta_em"],
            "additive_delta_dep_f1": base[str(l)]["delta_dep_f1"] + base[str(lp)]["delta_dep_f1"],
        }
    return out


def compare_with_impossible(
    single_results: Dict[str, dict],
    impossible_baseline: EvaluationResult,
) -> Dict[str, list]:
    """Δ_impossible per layer for the zero_full intervention."""
    base = single_results["zero_full"]
    out  = {"em": [], "dep_f1": []}
    for layer in sorted(int(k) for k in base.keys()):
        v = base[str(layer)]
        out["em"].append({
            "layer":    layer,
            "abs_diff": abs(v["em"] - impossible_baseline.em),
        })
        out["dep_f1"].append({
            "layer":    layer,
            "abs_diff": abs(v["dep_f1"] - impossible_baseline.dep_f1),
        })
    return out


def correlate_with_probing(
    single_results: Dict[str, dict],
    probing_path: Optional[str],
) -> Dict:
    """Correlate layer-wise ablation impact (zero_full) with probing divergence."""
    if not probing_path or not os.path.exists(probing_path):
        return {"available": False}
    if not _SCIPY_OK:
        return {"available": False, "reason": "scipy not installed"}
    with open(probing_path, "r") as fh:
        probing = json.load(fh)
    div = probing.get("divergence")
    if div is None:
        return {"available": False, "reason": "no divergence key in probing results"}

    n_layers = len(single_results["zero_full"])
    layer_probe: Dict[str, List[float]] = {}
    for prop in ("pos", "dep_rel", "depth"):
        prop_data = div.get(prop)
        if prop_data is None:
            layer_probe[prop] = [float("nan")] * n_layers
            continue
        vals = []
        for layer in range(n_layers):
            if layer < len(prop_data):
                row = prop_data[layer]
                if isinstance(row, list) and row:
                    vals.append(float(np.mean(row)))
                elif isinstance(row, dict) and row:
                    vals.append(float(np.mean(list(row.values()))))
                else:
                    vals.append(float("nan"))
            else:
                vals.append(float("nan"))
        layer_probe[prop] = vals

    base     = single_results["zero_full"]
    layers   = sorted(int(k) for k in base.keys())
    delta_em = [base[str(l)]["delta_em"]     for l in layers]
    delta_f1 = [base[str(l)]["delta_dep_f1"] for l in layers]

    out: Dict = {"available": True}
    for prop in ("pos", "dep_rel", "depth"):
        probe_arr = np.array(layer_probe[prop])
        mask = ~np.isnan(probe_arr)
        if mask.sum() < 3:
            continue
        rho_em, p_em = spearmanr(probe_arr[mask], np.array(delta_em)[mask])
        rho_f1, p_f1 = spearmanr(probe_arr[mask], np.array(delta_f1)[mask])
        out[f"{prop}_em_rho"]  = float(rho_em)
        out[f"{prop}_em_p"]    = float(p_em)
        out[f"{prop}_f1_rho"]  = float(rho_f1)
        out[f"{prop}_f1_p"]    = float(p_f1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> InterventionConfig:
    p = argparse.ArgumentParser(
        description="Causal intervention experiment for translator GPT-2."
    )
    p.add_argument("--translator_model", required=True)
    p.add_argument("--impossible_model", required=True)
    p.add_argument("--base_model", default="gpt2")
    p.add_argument("--dataset", required=True)
    p.add_argument("--probing_results", default="probing_results.json")
    p.add_argument("--output", default="causal_intervention_results.json")
    p.add_argument("--num_sentences", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    return InterventionConfig(
        translator_model=args.translator_model,
        impossible_model=args.impossible_model,
        base_model=args.base_model,
        dataset=args.dataset,
        output=args.output,
        probing_results=args.probing_results,
        num_sentences=args.num_sentences,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"Device: {cfg.device}")
    print(f"Loading dataset: {cfg.dataset}")
    pairs = load_dataset(cfg.dataset, cfg.num_sentences)
    print(f"  -> {len(pairs)} sentence pairs")

    # ------------------------------------------------------------------
    # Translator (fine-tuned; uses "Fix this text:" prompt format)
    # ------------------------------------------------------------------
    print(f"\nLoading translator: {cfg.translator_model}")
    translator_model, tokenizer = load_model(cfg.translator_model, cfg.device)
    translator = InterventionModel(
        translator_model, tokenizer, cfg.device, is_translator=True
    )

    print("Estimating activation statistics for mean/random interventions ...")
    translator.estimate_activation_statistics([p[0] for p in pairs[:50]])

    dep_eval = DependencyEvaluator()
    if dep_eval.nlp is None:
        print("  WARNING: spaCy en_core_web_sm not available; dep_f1/UAS/LAS = 0.")

    print("\nEvaluating intact translator baseline ...")
    translator.clear()
    baseline = evaluate(translator, pairs, dep_eval, cfg.max_new_tokens)
    print(
        f"  baseline  EM={baseline.em:.3f}  tokAcc={baseline.token_acc:.3f}  "
        f"BLEU1={baseline.bleu1:.3f}  depF1={baseline.dep_f1:.3f}"
    )

    # Sanity-check: show first prediction so prompt format can be verified.
    sample_pred = translator.generate([pairs[0][0]], max_new_tokens=cfg.max_new_tokens)[0]
    print(f"\n  [sanity] src : {pairs[0][0]}")
    print(f"  [sanity] ref : {pairs[0][1]}")
    print(f"  [sanity] pred: {sample_pred}\n")

    # ------------------------------------------------------------------
    # Single-layer ablations
    # ------------------------------------------------------------------
    print("Running single-layer ablations (12 layers x 3 kinds x 3 scopes) ...")
    single_results = run_single_layer_ablations(
        translator, pairs, dep_eval, cfg, baseline
    )

    # ------------------------------------------------------------------
    # Cumulative ablations
    # ------------------------------------------------------------------
    print("\nRunning cumulative ablations (early / late / middle windows) ...")
    cumulative_results = run_cumulative_ablations(
        translator, pairs, dep_eval, cfg, baseline
    )

    # ------------------------------------------------------------------
    # Interaction analysis
    # ------------------------------------------------------------------
    print("\nComputing pairwise additive baselines for interaction analysis ...")
    interaction_results = compute_interactions(single_results)

    # ------------------------------------------------------------------
    # Impossible model baseline (raw scrambled text — no translator prompt)
    # ------------------------------------------------------------------
    print(f"\nLoading impossible model: {cfg.impossible_model}")
    imp_model, imp_tok = load_model(cfg.impossible_model, cfg.device)
    impossible = InterventionModel(
        imp_model, imp_tok, cfg.device, is_translator=False
    )
    print("Evaluating impossible baseline ...")
    impossible_baseline = evaluate(impossible, pairs, dep_eval, cfg.max_new_tokens)
    print(
        f"  impossible  EM={impossible_baseline.em:.3f}  "
        f"depF1={impossible_baseline.dep_f1:.3f}"
    )

    # ------------------------------------------------------------------
    # Base GPT-2 baseline (raw scrambled text)
    # ------------------------------------------------------------------
    print(f"\nLoading base model: {cfg.base_model}")
    base_model, base_tok = load_model(cfg.base_model, cfg.device)
    base_wrapper = InterventionModel(
        base_model, base_tok, cfg.device, is_translator=False
    )
    print("Evaluating base GPT-2 baseline ...")
    base_baseline = evaluate(base_wrapper, pairs, dep_eval, cfg.max_new_tokens)
    print(
        f"  base GPT-2  EM={base_baseline.em:.3f}  depF1={base_baseline.dep_f1:.3f}"
    )

    # ------------------------------------------------------------------
    # Distance to impossible
    # ------------------------------------------------------------------
    print("\nComputing distance to impossible per ablated layer ...")
    distance_to_impossible = compare_with_impossible(single_results, impossible_baseline)

    # ------------------------------------------------------------------
    # Probing correlation
    # ------------------------------------------------------------------
    print("Correlating ablation impact with probing divergence ...")
    probing_corr = correlate_with_probing(single_results, cfg.probing_results)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = {
        "config": {
            "translator_model": cfg.translator_model,
            "impossible_model": cfg.impossible_model,
            "base_model":       cfg.base_model,
            "dataset":          cfg.dataset,
            "num_sentences":    cfg.num_sentences,
            "max_new_tokens":   cfg.max_new_tokens,
            "em_threshold":     cfg.em_threshold,
            "f1_threshold":     cfg.f1_threshold,
        },
        "baselines": {
            "translator": baseline.to_dict(),
            "impossible":  impossible_baseline.to_dict(),
            "base":        base_baseline.to_dict(),
        },
        "single_layer_ablations":      single_results,
        "cumulative_ablations":        cumulative_results,
        "pairwise_additive_baselines": interaction_results,
        "distance_to_impossible":      distance_to_impossible,
        "probing_correlation":         probing_corr,
    }

    print(f"\nWriting results to {cfg.output}")
    with open(cfg.output, "w") as fh:
        json.dump(output, fh, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
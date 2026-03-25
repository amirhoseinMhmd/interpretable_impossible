import torch
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats

import spacy


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

        # Register hooks to capture per-head outputs
        self._head_outputs = {}
        self._hooks = []
        self._register_hooks()

    # -- hook machinery -----------------------------------------------------

    def _register_hooks(self):
        """Attach forward hooks to every attention layer to capture
        per-head output **before** the output projection merges heads."""
        for layer_idx in range(self.n_layers):
            attn_module = self.model.transformer.h[layer_idx].attn
            hook = attn_module.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # GPT-2 attention forward returns (attn_output, present, (attentions))
            # attn_output shape: (batch, seq_len, n_embd)
            attn_output = output[0]  # (1, seq, n_embd)
            batch, seq, _ = attn_output.shape
            # Reshape into per-head representations
            # (1, seq, n_heads, d_head)
            per_head = attn_output.view(batch, seq, self.n_heads, self.d_head)
            self._head_outputs[layer_idx] = per_head.detach().cpu()
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- extraction ---------------------------------------------------------

    def extract(self, text: str):

        self._head_outputs.clear()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        seq_len = inputs["input_ids"].shape[1]
        if seq_len == 0:
            return None, [], []

        with torch.no_grad():
            self.model(**inputs)

        token_ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode(tid) for tid in token_ids]

        head_reps = {}
        for layer_idx in range(self.n_layers):
            per_head = self._head_outputs[layer_idx][0]  # (seq, n_heads, d_head)
            for head_idx in range(self.n_heads):
                head_reps[(layer_idx, head_idx)] = per_head[:, head_idx, :].numpy()

        return head_reps, token_ids, tokens


# ---------------------------------------------------------------------------
# Subword-to-word alignment
# ---------------------------------------------------------------------------

def align_subwords_to_words(subword_tokens: list[str], word_labels: dict):

    words = word_labels["tokens"]
    aligned = []
    sw_idx = 0

    for w_idx, word in enumerate(words):
        # Build up the word from subword tokens
        accumulated = ""
        start_sw = sw_idx
        while sw_idx < len(subword_tokens):
            piece = subword_tokens[sw_idx].replace("Ġ", "").replace(" ", "")
            accumulated += piece
            sw_idx += 1
            # Check if we've reconstructed the word (strip punctuation quirks)
            if accumulated == word.replace(" ", ""):
                break
            # If accumulated is already longer than the word, alignment failed
            if len(accumulated) > len(word) + 2:
                # fallback: skip this word
                sw_idx = start_sw + 1
                break

        if accumulated == word.replace(" ", ""):
            aligned.append({
                "subword_idx": start_sw,
                "pos": word_labels["pos"][w_idx],
                "dep_rel": word_labels["dep_rel"][w_idx],
                "head_idx": word_labels["head_idx"][w_idx],
                "depth": word_labels["depth"][w_idx],
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
        (l, h): {"X": [], "pos": [], "dep_rel": [], "depth": [], "head_idx": []}
        for l in range(n_layers)
        for h in range(n_heads)
    }

    skipped = 0
    for i, (scrambled, original) in enumerate(
        zip(scrambled_sentences, original_sentences)
    ):
        if i % 100 == 0:
            print(f"  Building probing data: {i}/{len(scrambled_sentences)}")

        # Gold labels from original sentence
        labels = labeler.extract_labels(original)

        # Model representations from scrambled sentence
        head_reps, token_ids, tokens = extractor.extract(scrambled)
        if head_reps is None:
            skipped += 1
            continue

        # Align subwords to words
        aligned = align_subwords_to_words(tokens, labels)
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

    if skipped:
        print(f"  Skipped {skipped}/{len(scrambled_sentences)} sentences (alignment)")

    # Convert lists to arrays
    for key in dataset:
        if len(dataset[key]["X"]) > 0:
            dataset[key]["X"] = np.stack(dataset[key]["X"])
            dataset[key]["depth"] = np.array(dataset[key]["depth"], dtype=np.float32)
        else:
            dataset[key]["X"] = np.empty((0, extractor.d_head))
            dataset[key]["depth"] = np.empty(0)

    return dataset


# ---------------------------------------------------------------------------
# Probing classifiers
# ---------------------------------------------------------------------------

class ProbingExperiment:
    """Train and evaluate linear probes for POS, dependency, and depth."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    # -- NaN sanitiser ------------------------------------------------------

    @staticmethod
    def _sanitise(X: np.ndarray, y: np.ndarray):
        """Drop rows where any feature is NaN or Inf.

        Returns (X_clean, y_clean) or (None, None) if too few rows survive.
        """
        finite_mask = np.isfinite(X).all(axis=1)
        n_bad = (~finite_mask).sum()
        if n_bad:
            print(f"    [sanitise] dropping {n_bad} rows with NaN/Inf features")
        X_c, y_c = X[finite_mask], y[finite_mask]
        if len(X_c) < 20:
            return None, None
        return X_c, y_c

    # -- Safe split helper --------------------------------------------------

    def _safe_train_test_split(self, X: np.ndarray, y: np.ndarray, n_classes: int):
        """Stratified split when possible; drops singleton classes otherwise.

        Returns (X_train, X_test, y_train, y_test) or None if too few samples
        remain after filtering rare classes.
        """
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

    def probe_pos(self, X: np.ndarray, labels: list[str]):
        """Linear probe for POS classification.

        Returns dict with accuracy, f1, and baseline scores.
        """
        if len(X) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X, y = self._sanitise(X, y)
        if X is None:
            return None

        n_classes = len(np.unique(y))

        split = self._safe_train_test_split(X, y, n_classes)
        if split is None:
            return None
        X_train, X_test, y_train, y_test = split

        # Linear probe
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=self.random_state,
        )
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
        clf_rand = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=self.random_state,
        )
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

    def probe_dependency(self, X: np.ndarray, labels: list[str]):
        """Linear probe for dependency relation classification."""
        if len(X) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X, y = self._sanitise(X, y)
        if X is None:
            return None

        n_classes = len(np.unique(y))

        split = self._safe_train_test_split(X, y, n_classes)
        if split is None:
            return None
        X_train, X_test, y_train, y_test = split

        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=self.random_state,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        random_acc = 1.0 / n_classes
        majority_class = np.bincount(y_train).argmax()
        majority_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)
        clf_rand = LogisticRegression(
            max_iter=1000, solver="lbfgs",
            random_state=self.random_state,
        )
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

    def probe_depth(self, X: np.ndarray, depths: np.ndarray):
        """Linear probe for syntactic depth (regression)."""
        if len(X) < 20:
            return None

        X, depths = self._sanitise(X, depths)
        if X is None:
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X, depths, test_size=self.test_size, random_state=self.random_state,
        )

        reg = Ridge(alpha=1.0)
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
        reg_rand = Ridge(alpha=1.0)
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
):
    """Train probes on word embeddings (wte) as a control baseline."""
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]

    X_all, pos_all, dep_all, depth_all = [], [], [], []
    wte = extractor.model.transformer.wte  # word token embedding layer

    for i, (scrambled, original) in enumerate(
        zip(scrambled_sentences, original_sentences)
    ):
        if i % 100 == 0:
            print(f"  Word-embedding baseline: {i}/{len(scrambled_sentences)}")

        labels = labeler.extract_labels(original)
        inputs = extractor.tokenizer(
            scrambled, return_tensors="pt", truncation=True, max_length=512
        ).to(extractor.device)

        if inputs["input_ids"].shape[1] == 0:
            continue

        with torch.no_grad():
            embeddings = wte(inputs["input_ids"])[0].cpu().numpy()  # (seq, emb)

        tokens = [
            extractor.tokenizer.decode(tid)
            for tid in inputs["input_ids"][0].tolist()
        ]
        aligned = align_subwords_to_words(tokens, labels)

        for entry in aligned:
            sw_idx = entry["subword_idx"]
            if sw_idx < embeddings.shape[0]:
                X_all.append(embeddings[sw_idx])
                pos_all.append(entry["pos"])
                dep_all.append(entry["dep_rel"])
                depth_all.append(entry["depth"])

    if len(X_all) < 20:
        return {"pos": None, "dep_rel": None, "depth": None}

    X_all = np.stack(X_all)
    depth_all = np.array(depth_all, dtype=np.float32)

    prober = ProbingExperiment(test_size=test_size, random_state=random_state)
    return {
        "pos": prober.probe_pos(X_all, pos_all),
        "dep_rel": prober.probe_dependency(X_all, dep_all),
        "depth": prober.probe_depth(X_all, depth_all),
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

    # Word-embedding baseline
    print("\n  Computing word-embedding baseline...")
    emb_baseline = compute_word_embedding_baseline(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
    )

    # Probe each head
    prober = ProbingExperiment()
    n_layers = extractor.n_layers
    n_heads = extractor.n_heads

    per_head = {}
    total = n_layers * n_heads
    done = 0

    for l in range(n_layers):
        for h in range(n_heads):
            done += 1
            if done % 24 == 0:
                print(f"  Probing head {done}/{total}")

            data = dataset[(l, h)]
            X = data["X"]
            if X.shape[0] < 20:
                per_head[(l, h)] = {"pos": None, "dep_rel": None, "depth": None}
                continue

            pos_result = prober.probe_pos(X, data["pos"])
            dep_result = prober.probe_dependency(X, data["dep_rel"])
            depth_result = prober.probe_depth(X, data["depth"])

            per_head[(l, h)] = {
                "pos": pos_result,
                "dep_rel": dep_result,
                "depth": depth_result,
            }

    # Layer-wise summary
    layer_summary = _compute_layer_summary(per_head, n_layers, n_heads)

    # Clean up
    extractor.remove_hooks()
    del extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "per_head": per_head,
        "layer_summary": layer_summary,
        "word_embedding_baseline": emb_baseline,
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


# ---------------------------------------------------------------------------
# Divergence analysis (translator vs impossible)
# ---------------------------------------------------------------------------

def compute_probing_divergence(results_translator, results_impossible, n_layers, n_heads):
    divergence = {}
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

    # Top heads per property
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


def print_divergence_results(divergence, correlations):
    print(f"\n{'='*60}")
    print(f"  Probing Divergence (Translator - Impossible)")
    print(f"{'='*60}")
    for prop, delta in divergence.items():
        metric = "ΔAccuracy" if prop != "depth" else "ΔSpearman_r"
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
    )
    print_probing_results(results_impossible, "Impossible")

    # ---- Divergence analysis ----
    # Determine grid size from first model
    sample_key = next(iter(results_translator["per_head"]))
    n_layers = max(k[0] for k in results_translator["per_head"]) + 1
    n_heads = max(k[1] for k in results_translator["per_head"]) + 1

    divergence = compute_probing_divergence(
        results_translator, results_impossible, n_layers, n_heads
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
        },
        "tokenizer": tokenizer_path,
        "n_sentences": len(scrambled_sentences),
        "translator": _serialise_results(results_translator),
        "impossible": _serialise_results(results_impossible),
        "divergence": {prop: delta.tolist() for prop, delta in divergence.items()},
        "entropy_probing_correlation": correlations,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


# python prob_classifier.py \
#   --translation_model models/gutenberg-localShuffle-w3 \
#   --impossible_model mission-impossible-lms/local-shuffle-w3-gpt2 \
#   --dataset test_data/training_data_1k_gutenberg_localShuffle.json \
#   --entropy_results entropy_impossible_results.json \
#   --output probing_results.json \
#   --max_sentences 200
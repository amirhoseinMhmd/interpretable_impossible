import argparse
import json

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
        print(f"Using device: {self.device}", flush=True)

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
                arr = per_head[:, head_idx, :].numpy()
                # Replace NaN/Inf at source to avoid downstream data loss
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                head_reps[(layer_idx, head_idx)] = arr

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
                "word_idx": w_idx,              # original word position
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
            print(f"  Building probing data: {i}/{len(scrambled_sentences)}", flush=True)

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
        print(
            f"  Skipped {skipped}/{len(scrambled_sentences)} sentences (alignment)",
            flush=True,
        )

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
            print(f"    [sanitise] dropping {n_bad} rows with NaN/Inf features", flush=True)
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

        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
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
        reg_rand = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
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
    ):
        assert combination in self.COMBINATION_METHODS
        self.combination = combination
        self.test_size = test_size
        self.random_state = random_state
        self.max_pairs_per_sentence = max_pairs_per_sentence

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
    def _sanitise(X, y):
        finite_mask = np.isfinite(X).all(axis=1)
        n_bad = (~finite_mask).sum()
        if n_bad:
            print(f"    [pairwise sanitise] dropping {n_bad} rows with NaN/Inf", flush=True)
        X_c, y_c = X[finite_mask], y[finite_mask]
        if len(X_c) < 20:
            return None, None
        return X_c, y_c

    # -- Binary arc prediction ----------------------------------------------

    def probe_arc(self, X_pairs, y_arc):
        """Binary classifier: does token i depend on token j?

        Returns dict with accuracy, precision, recall, F1, and baselines.
        """
        if len(X_pairs) < 20:
            return None

        X_pairs, y_arc = self._sanitise(X_pairs, y_arc)
        if X_pairs is None:
            return None

        n_pos = int(y_arc.sum())
        n_neg = len(y_arc) - n_pos
        if n_pos < 2 or n_neg < 2:
            return None

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

    def probe_relation(self, X_pairs, y_rel_labels):
        """Multi-class classifier for dependency relation type.

        Trained only on positive pairs (where a dependency arc exists).
        Returns dict with accuracy, macro F1, and baselines.
        """
        if len(X_pairs) < 20:
            return None

        le = LabelEncoder()
        y = le.fit_transform(y_rel_labels)

        X_pairs, y = self._sanitise(X_pairs, y)
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
            if len(X_pairs) < 20:
                return None
            n_classes = len(np.unique(y))

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
):
    """Build pairwise token-pair dataset for dependency structure probing.

    For each sentence, constructs all ordered pairs (i, j) where i != j,
    with gold labels from the original parse.

    Returns dict keyed by (layer, head) with:
        X_pairs: combined representations c_ij
        y_arc: binary arc labels
        y_rel: relation labels (for positive pairs)
        distances: |i - j| for distance baseline
    """
    if max_sentences:
        scrambled_sentences = scrambled_sentences[:max_sentences]
        original_sentences = original_sentences[:max_sentences]

    n_layers = extractor.n_layers
    n_heads = extractor.n_heads
    combiner = PairwiseDependencyProber(combination=combination)

    # Collect per-sentence pairwise data, then combine per-head
    # To save memory, accumulate X for one (layer, head) at a time is impractical
    # Instead, collect aligned sentence data first, then build pairs per head

    sentence_data = []  # list of dicts with aligned info and head_reps

    skipped = 0
    for i, (scrambled, original) in enumerate(
        zip(scrambled_sentences, original_sentences)
    ):
        if i % 100 == 0:
            print(f"  Building pairwise data: {i}/{len(scrambled_sentences)}", flush=True)

        labels = labeler.extract_labels(original)
        head_reps, token_ids, tokens = extractor.extract(scrambled)
        if head_reps is None:
            skipped += 1
            continue

        aligned = align_subwords_to_words(tokens, labels)
        if len(aligned) < 2:  # need at least 2 tokens for pairs
            skipped += 1
            continue

        sentence_data.append({
            "aligned": aligned,
            "head_reps": head_reps,
            "n_aligned": len(aligned),
        })

    if skipped:
        print(
            f"  Skipped {skipped}/{len(scrambled_sentences)} sentences (pairwise alignment)",
            flush=True,
        )

    # Now build per-head pairwise datasets
    dataset = {}
    for l in range(n_layers):
        for h in range(n_heads):
            X_pairs_list = []
            y_arc_list = []
            y_rel_pos_list = []    # relation labels for positive pairs only
            X_pairs_pos_list = []  # representations for positive pairs only
            distances_list = []

            for sent in sentence_data:
                aligned = sent["aligned"]
                n_tok = sent["n_aligned"]
                reps = sent["head_reps"][(l, h)]  # (seq, d_head)

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

                        # Gold: does token at word position idx_i depend on
                        # word at position idx_j?
                        head_of_i = aligned[idx_i]["head_idx"]
                        # head_idx is the word index in the original sentence
                        # We need to check if idx_j's original word position
                        # matches head_of_i. Since aligned preserves word order,
                        # idx_j corresponds to the j-th aligned word.
                        # But head_idx is the absolute index in the spaCy doc.
                        # We need a mapping from aligned index -> original word idx.
                        # aligned entries are in word order, so aligned[k] is the
                        # k-th successfully aligned word. Its original word index
                        # is implicit from the alignment loop in
                        # align_subwords_to_words. We need that index.

                        # For now, use the head_idx directly: arc exists if
                        # aligned[idx_i].head_idx points to a word whose
                        # subword_idx matches aligned[idx_j].subword_idx position.
                        # This is approximate — we store original word positions
                        # in the next step.

                        is_arc = 0
                        rel_label = "NO_ARC"

                        # head_of_i is the original word index of i's head.
                        # We check all aligned tokens to find if idx_j maps to
                        # that original word.  Since we don't store original
                        # word indices in aligned, we use a heuristic: if
                        # head_of_i == idx_j (only correct if no words were
                        # skipped in alignment).
                        # This is a limitation we fix below by storing word_idx.
                        if head_of_i == aligned[idx_j].get("word_idx", idx_j):
                            is_arc = 1
                            rel_label = aligned[idx_i]["dep_rel"]

                        X_pairs_list.append(c_ij)
                        y_arc_list.append(is_arc)
                        distances_list.append(abs(sw_i - sw_j))

                        if is_arc:
                            X_pairs_pos_list.append(c_ij)
                            y_rel_pos_list.append(rel_label)

            if len(X_pairs_list) > 0:
                dataset[(l, h)] = {
                    "X_pairs": np.stack(X_pairs_list),
                    "y_arc": np.array(y_arc_list, dtype=np.int32),
                    "X_pairs_pos": np.stack(X_pairs_pos_list) if X_pairs_pos_list else np.empty((0, X_pairs_list[0].shape[0])),
                    "y_rel_pos": y_rel_pos_list,
                    "distances": np.array(distances_list, dtype=np.int32),
                }
            else:
                d_comb = extractor.d_head * (2 if combination == "concat" else
                                              1 if combination in ("diff", "product") else 4)
                dataset[(l, h)] = {
                    "X_pairs": np.empty((0, d_comb)),
                    "y_arc": np.empty(0, dtype=np.int32),
                    "X_pairs_pos": np.empty((0, d_comb)),
                    "y_rel_pos": [],
                    "distances": np.empty(0, dtype=np.int32),
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

    for i, (scrambled, original) in enumerate(
        zip(scrambled_sentences, original_sentences)
    ):
        if i % 100 == 0:
            print(f"  Pairwise baseline: {i}/{len(scrambled_sentences)}", flush=True)

        labels = labeler.extract_labels(original)
        inputs = extractor.tokenizer(
            scrambled, return_tensors="pt", truncation=True, max_length=512
        ).to(extractor.device)

        if inputs["input_ids"].shape[1] == 0:
            continue

        with torch.no_grad():
            embeddings = wte(inputs["input_ids"])[0].cpu().numpy()

        tokens = [
            extractor.tokenizer.decode(tid)
            for tid in inputs["input_ids"][0].tolist()
        ]
        aligned = align_subwords_to_words(tokens, labels)
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
                distances_all.append(abs(sw_i - sw_j))

                if is_arc:
                    X_emb_pos.append(c_ij)
                    y_rel_pos.append(rel_label)

    result = {"word_emb_arc": None, "word_emb_rel": None, "distance_arc": None}

    if len(X_emb_pairs) < 20:
        return result

    X_emb_pairs = np.stack(X_emb_pairs)
    y_arc_all = np.array(y_arc_all, dtype=np.int32)
    distances_all = np.array(distances_all, dtype=np.int32)

    prober = PairwiseDependencyProber(
        combination=combination, test_size=test_size, random_state=random_state
    )

    # Word-embedding arc baseline
    result["word_emb_arc"] = prober.probe_arc(X_emb_pairs, y_arc_all)

    # Word-embedding relation baseline (positive pairs only)
    if len(X_emb_pos) >= 20:
        X_emb_pos = np.stack(X_emb_pos)
        result["word_emb_rel"] = prober.probe_relation(X_emb_pos, y_rel_pos)

    # Distance baseline: predict arc from |i-j| alone
    dist_features = distances_all.reshape(-1, 1).astype(np.float32)
    result["distance_arc"] = prober.probe_arc(dist_features, y_arc_all)

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
            print(f"  Word-embedding baseline: {i}/{len(scrambled_sentences)}", flush=True)

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

    print(f"\n{'='*60}", flush=True)
    print(f"  Probing: {model_label} ({model_path})", flush=True)
    print(f"{'='*60}", flush=True)

    extractor = HeadRepresentationExtractor(
        model_path, tokenizer_path=tokenizer_path, device=device,
    )

    # Build dataset
    print("\n  Building probing dataset...", flush=True)
    dataset = build_probing_dataset(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
    )

    # Word-embedding baseline (token-level)
    print("\n  Computing word-embedding baseline...", flush=True)
    emb_baseline = compute_word_embedding_baseline(
        extractor, labeler, scrambled_sentences, original_sentences,
        max_sentences=max_sentences,
    )

    # Probe each head (token-level)
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
                print(f"  Probing head {done}/{total}", flush=True)

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

    # ---- Pairwise dependency structure probing ----
    print("\n  Building pairwise probing dataset...", flush=True)
    pw_dataset = build_pairwise_dataset(
        extractor, labeler, scrambled_sentences, original_sentences,
        combination="concat",
        max_sentences=max_sentences,
    )

    print("\n  Computing pairwise baselines...", flush=True)
    pw_baselines = compute_pairwise_baselines(
        extractor, labeler, scrambled_sentences, original_sentences,
        combination="concat",
        max_sentences=max_sentences,
    )

    pw_prober = PairwiseDependencyProber(combination="concat")
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

            arc_result = pw_prober.probe_arc(X_pairs, pw_data["y_arc"])
            rel_result = pw_prober.probe_relation(
                pw_data["X_pairs_pos"], pw_data["y_rel_pos"],
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
    print(f"Loaded {len(scrambled_sentences)} sentence pairs", flush=True)

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

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse


class AttentionEntropyAnalyzer:

    def __init__(self, model_path, tokenizer_path=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        tok_path = tokenizer_path if tokenizer_path else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="eager",
        )
        self.model.to(self.device)
        self.model.eval()

        self.n_layers = self.model.config.n_layer
        self.n_heads = self.model.config.n_head
        self.model_path = model_path

    def extract_attention_weights(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        if inputs["input_ids"].shape[1] == 0:
            return None, 0

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        if outputs.attentions is None:
            return None, 0

        attentions = [attn.squeeze(0).cpu() for attn in outputs.attentions]
        seq_len = inputs["input_ids"].shape[1]
        return attentions, seq_len

    @staticmethod
    def compute_entropy(attention_probs):
        p = attention_probs.float()
        p = p[p > 0]
        log_p = torch.log2(p)
        entropy = -(p * log_p).sum().item()
        return entropy

    def compute_per_head_entropy(self, text):
        attentions, seq_len = self.extract_attention_weights(text)

        if attentions is None:
            return None

        entropy_matrix = np.zeros((self.n_layers, self.n_heads))
        for layer_idx, attn in enumerate(attentions):
            for head_idx in range(self.n_heads):
                token_entropies = []
                for query_pos in range(seq_len):
                    h = self.compute_entropy(attn[head_idx, query_pos, :])
                    token_entropies.append(h)
                entropy_matrix[layer_idx, head_idx] = np.mean(token_entropies)

        return entropy_matrix

    def analyze_dataset(self, sentences):
        all_entropies = []
        skipped = 0
        for i, sent in enumerate(sentences):
            if i % 50 == 0:
                print(f"  Processing sentence {i}/{len(sentences)}")
            ent = self.compute_per_head_entropy(sent)
            if ent is not None:
                all_entropies.append(ent)
            else:
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} empty/unparseable sentences")

        if len(all_entropies) == 0:
            raise ValueError("All sentences were skipped. Check tokenizer compatibility.")

        return np.mean(all_entropies, axis=0)


def load_dataset(dataset_path, max_sentences=None):
    with open(dataset_path, "r") as f:
        pairs = json.load(f)

    if max_sentences:
        pairs = pairs[:max_sentences]

    possible_sentences = [pair[0] for pair in pairs]
    impossible_sentences = [pair[1] for pair in pairs]

    return possible_sentences, impossible_sentences


def compute_pairwise_divergence(H_a, H_b):
    """Compute ΔH = H_b - H_a for each head."""
    return H_b - H_a


def identify_divergent_heads(delta_H, threshold_bits=0.5):
    """Identify heads with |ΔH| > threshold, separating positive and negative."""
    n_layers, n_heads = delta_H.shape
    positive_heads = []
    negative_heads = []

    for layer in range(n_layers):
        for head in range(n_heads):
            val = float(delta_H[layer, head])
            if val > threshold_bits:
                positive_heads.append({"layer": layer, "head": head, "delta_H": val})
            elif val < -threshold_bits:
                negative_heads.append({"layer": layer, "head": head, "delta_H": val})

    positive_heads.sort(key=lambda x: x["delta_H"], reverse=True)
    negative_heads.sort(key=lambda x: x["delta_H"])

    return positive_heads, negative_heads


def compute_layer_summary(delta_H):
    n_layers = delta_H.shape[0]
    layer_means = [float(np.mean(delta_H[l, :])) for l in range(n_layers)]
    layer_maxes = [float(np.max(delta_H[l, :])) for l in range(n_layers)]
    layer_mins = [float(np.min(delta_H[l, :])) for l in range(n_layers)]
    peak_layer = int(np.argmax(layer_means))

    return {
        "layer_mean_delta_H": layer_means,
        "layer_max_delta_H": layer_maxes,
        "layer_min_delta_H": layer_mins,
        "peak_layer": peak_layer,
        "peak_layer_mean_delta_H": layer_means[peak_layer],
    }


def print_pair_results(pair_name, delta_H, pos_heads, neg_heads, layer_summary):
    print(f"\n{'='*60}")
    print(f"  {pair_name}")
    print(f"{'='*60}")
    print(f"  Mean ΔH: {np.mean(delta_H):.3f} bits")
    print(f"  Max ΔH:  {np.max(delta_H):.3f} bits")
    print(f"  Min ΔH:  {np.min(delta_H):.3f} bits")
    print(f"  Peak layer: {layer_summary['peak_layer']}")
    print(f"\n  Positive ΔH heads (model_B more focused): {len(pos_heads)}")
    for h in pos_heads[:5]:
        print(f"    Layer {h['layer']}, Head {h['head']}: ΔH = {h['delta_H']:.3f}")
    print(f"\n  Negative ΔH heads (model_A more focused): {len(neg_heads)}")
    for h in neg_heads[:5]:
        print(f"    Layer {h['layer']}, Head {h['head']}: ΔH = {h['delta_H']:.3f}")

    print(f"\n  Layer-wise mean ΔH:")
    n_layers = len(layer_summary["layer_mean_delta_H"])
    for l in range(n_layers):
        val = layer_summary["layer_mean_delta_H"][l]
        bar = "▓" * int(abs(val) * 10)
        sign = "+" if val >= 0 else "-"
        print(f"    Layer {l:2d}: {val:+.3f} {sign}{bar}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Three-way attention entropy comparison: translation vs impossible vs normal GPT-2"
    )

    parser.add_argument(
        "--translation_model", type=str, required=True,
        help="Path or HuggingFace ID for the translation model",
    )
    parser.add_argument(
        "--impossible_model", type=str, required=True,
        help="Path or HuggingFace ID for Kallini's impossible model",
    )
    parser.add_argument(
        "--normal_model", type=str, default="gpt2",
        help="Path or HuggingFace ID for normal GPT-2 (default: gpt2)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Shared tokenizer (default: use impossible model's tokenizer)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to JSON dataset: [[possible, impossible], ...]",
    )
    parser.add_argument(
        "--perturbation_name", type=str, required=True,
        choices=["LOCALSHUFFLE", "PARTIALREVERSE", "WORDHOP"],
    )
    parser.add_argument(
        "--input_type", type=str, default="impossible",
        choices=["impossible", "possible"],
        help="Which sentences to feed to all three models (default: impossible)",
    )
    parser.add_argument(
        "--output", type=str, default="rq1_three_way_results.json",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
    )
    parser.add_argument(
        "--max_sentences", type=int, default=None,
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "mps", "cpu"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device) if args.device else None
    tokenizer_path = args.tokenizer if args.tokenizer else args.impossible_model

    # Load dataset
    possible_sentences, impossible_sentences = load_dataset(
        args.dataset, args.max_sentences
    )

    sentences = impossible_sentences if args.input_type == "impossible" else possible_sentences

    print(f"Perturbation: {args.perturbation_name}")
    print(f"Translation model: {args.translation_model}")
    print(f"Impossible model: {args.impossible_model}")
    print(f"Normal model: {args.normal_model}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Input type: {args.input_type} ({len(sentences)} sentences)")
    print(f"Device: {device or 'auto-detect'}")

    # Compute entropy for all three models
    print("\n=== Analyzing Translation Model ===")
    trans_analyzer = AttentionEntropyAnalyzer(
        args.translation_model, tokenizer_path=tokenizer_path, device=device
    )
    H_translation = trans_analyzer.analyze_dataset(sentences)
    del trans_analyzer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n=== Analyzing Impossible Model (Kallini) ===")
    imp_analyzer = AttentionEntropyAnalyzer(
        args.impossible_model, tokenizer_path=tokenizer_path, device=device
    )
    H_impossible = imp_analyzer.analyze_dataset(sentences)
    del imp_analyzer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n=== Analyzing Normal GPT-2 ===")
    norm_analyzer = AttentionEntropyAnalyzer(
        args.normal_model, tokenizer_path=tokenizer_path, device=device
    )
    H_normal = norm_analyzer.analyze_dataset(sentences)
    del norm_analyzer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Three pairwise comparisons
    pairs = {
        "translation_vs_impossible": {
            "delta_H": compute_pairwise_divergence(H_translation, H_impossible),
            "description": "ΔH = H_impossible - H_translation (positive = translation more focused)",
        },
        "translation_vs_normal": {
            "delta_H": compute_pairwise_divergence(H_translation, H_normal),
            "description": "ΔH = H_normal - H_translation (positive = translation more focused)",
        },
        "normal_vs_impossible": {
            "delta_H": compute_pairwise_divergence(H_normal, H_impossible),
            "description": "ΔH = H_impossible - H_normal (positive = normal more focused)",
        },
    }

    # Analyze each pair
    results = {
        "perturbation": args.perturbation_name,
        "input_type": args.input_type,
        "tokenizer": tokenizer_path,
        "models": {
            "translation": args.translation_model,
            "impossible": args.impossible_model,
            "normal": args.normal_model,
        },
        "raw_entropy": {
            "H_translation": H_translation.tolist(),
            "H_impossible": H_impossible.tolist(),
            "H_normal": H_normal.tolist(),
        },
        "comparisons": {},
    }

    for pair_name, pair_data in pairs.items():
        delta_H = pair_data["delta_H"]
        pos_heads, neg_heads = identify_divergent_heads(delta_H, args.threshold)
        layer_summary = compute_layer_summary(delta_H)

        print_pair_results(pair_name, delta_H, pos_heads, neg_heads, layer_summary)

        results["comparisons"][pair_name] = {
            "description": pair_data["description"],
            "delta_H": delta_H.tolist(),
            "positive_heads": pos_heads,
            "negative_heads": neg_heads,
            "num_positive_heads": len(pos_heads),
            "num_negative_heads": len(neg_heads),
            "mean_delta_H": float(np.mean(delta_H)),
            "max_delta_H": float(np.max(delta_H)),
            "min_delta_H": float(np.min(delta_H)),
            "layer_summary": layer_summary,
        }

    # Triangulation: find heads unique to each comparison
    def head_set(heads_list):
        return {(h["layer"], h["head"]) for h in heads_list}

    trans_vs_imp_pos = head_set(results["comparisons"]["translation_vs_impossible"]["positive_heads"])
    trans_vs_norm_pos = head_set(results["comparisons"]["translation_vs_normal"]["positive_heads"])
    norm_vs_imp_pos = head_set(results["comparisons"]["normal_vs_impossible"]["positive_heads"])

    # Heads where translation is more focused than BOTH other models
    translation_specific = trans_vs_imp_pos & trans_vs_norm_pos
    # Heads where the difference is only between impossible and normal (not translation-specific)
    impossible_vs_normal_only = norm_vs_imp_pos - trans_vs_imp_pos

    results["triangulation"] = {
        "translation_specific_heads": [
            {"layer": l, "head": h} for l, h in sorted(translation_specific)
        ],
        "impossible_vs_normal_only_heads": [
            {"layer": l, "head": h} for l, h in sorted(impossible_vs_normal_only)
        ],
    }

    print(f"\n{'='*60}")
    print(f"  TRIANGULATION")
    print(f"{'='*60}")
    print(f"\n  Translation-specific heads (more focused than BOTH impossible and normal):")
    if translation_specific:
        for l, h in sorted(translation_specific):
            dh1 = results["comparisons"]["translation_vs_impossible"]["delta_H"][l][h]
            dh2 = results["comparisons"]["translation_vs_normal"]["delta_H"][l][h]
            print(f"    Layer {l}, Head {h}: ΔH vs impossible = {dh1:.3f}, ΔH vs normal = {dh2:.3f}")
    else:
        print(f"    None found at threshold {args.threshold}")

    print(f"\n  Impossible-vs-normal-only heads (not translation-specific):")
    if impossible_vs_normal_only:
        for l, h in sorted(impossible_vs_normal_only):
            dh = results["comparisons"]["normal_vs_impossible"]["delta_H"][l][h]
            print(f"    Layer {l}, Head {h}: ΔH = {dh:.3f}")
    else:
        print(f"    None found at threshold {args.threshold}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
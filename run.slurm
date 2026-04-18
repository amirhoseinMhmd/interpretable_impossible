#SBATCH --job-name=interpretability
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=50:00:00

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1

source .venv/bin/activate


python attention_entropy.py --translation_model the-amirhosein/gutenberg-fullshuffle-s57 --impossible_model mission-impossible-lms/deterministic-shuffle-s57-gpt2 --dataset test_data/test_data_1k_gutenberg_fullShuffle.json --perturbation_name fullShuffle --input_type impossible  --output entropy_impossible_fullShuffle_results.json > entropy_impossible_fullShuffle.log

python prob_classifier.py --translation_model the-amirhosein/gutenberg-fullshuffle-s57 --impossible_model mission-impossible-lms/deterministic-shuffle-s57-gpt2 --dataset test_data/test_data_1k_gutenberg_fullShuffle.json --entropy_results entropy_impossible_fullShuffle_results.json --output probing_fullShuffle_results.json > probing_fullShuffle.log

python causal_intervention.py   --translator_model the-amirhosein/gutenberg-fullshuffle-s57   --impossible_model mission-impossible-lms/deterministic-shuffle-s57-gpt2   --base_model gpt2   --dataset test_data/test_data_1k_gutenberg_fullShuffle.json   --num_sentences 1000  --output causal_intervention_fullShuffle_results.json > causal_intervention_fullShuffle.log


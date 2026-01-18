from bpe import BPE
from utils import *
from markov_chain import *

import optuna


corpus = load_corpus('data/war_and_peace.txt')
text = preprocess_text(corpus)
train_text = text[:int(len(text) * 0.2)]
test_text = text[int(len(text) * 0.2):int(len(text) * 0.3)]

kernels = {
    "gaussian_kernel": gaussian_kernel,
    "triangular_kernel": triangular_kernel,
    "cosine_kernel": cosine_kernel,
    "exponential_kernel": exponential_kernel
}


def objective(trial):
    global train_text, test_text, kernels

    n = trial.suggest_int("n", 1, 7)
    num_merges = trial.suggest_int("num_merges", 0, 2000)
    smoothing = trial.suggest_categorical("smoothing", ["none", "add-k", "interpolation"])
    alpha = 0.0
    if smoothing == "add-k":
        alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
    
    kernel = trial.suggest_categorical("kernel", ["gaussian_kernel", "triangular_kernel", "cosine_kernel", "exponential_kernel"])

    bpe = BPE(merges=num_merges)

    tokens = bpe.encode(" ".join(train_text))

    train_tokens = tokens[:int(len(tokens) * 0.9)]
    test_tokens = tokens[int(len(tokens) * 0.9):]

    model = MarkovChain(n=n, smoothing=smoothing, alpha=alpha, kernel=kernels[kernel])
    model.train(train_tokens)

    gen_tokens = model.generate(max_length=500)

    gen_tokens = [int(i) for i in gen_tokens]
    gen_tokens = bpe.decode(gen_tokens)

    # print(f"n={n}, merges={num_merges}, alpha={alpha:.2e}")
    # print(f"train_tokens len: {len(train_tokens)}")
    # print(f"test_tokens len: {len(test_tokens)}")
    # print(f"Unique tokens in train: {len(set(train_tokens))}")

    # # Проверь perplexity вручную
    # ppl = calculate_perplexity(model, test_tokens)
    # print(f"Perplexity: {ppl:.2f}")
    # if ppl > 1e6:
    #     print("→ Too high! Likely due to no smoothing or large n.")

    score = count_score(model, test_tokens, gen_tokens.split(), train_text, test_text)

    return score


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение метрики:", study.best_value)
from bpe import BPE
from utils import *
from markov_chain import *

import optuna


corpus = load_corpus('data/war_and_peace.txt')
text = preprocess_text(corpus)
text = text[:int(len(text) * 0.1)]

kernels = {
    "gaussian_kernel": gaussian_kernel,
    "triangular_kernel": triangular_kernel,
    "cosine_kernel": cosine_kernel,
    "exponential_kernel": exponential_kernel
}


def objective(trial):
    global text, kernels

    n = trial.suggest_int("n", 1, 7)
    num_merges = trial.suggest_int("num_merges", 0, 2000)
    smoothing = trial.suggest_categorical("smoothing", ["none", "add-k", "interpolation"])
    alpha = 0.0
    if smoothing == "add-k":
        alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)
    
    kernel = trial.suggest_categorical("kernel", ["gaussian_kernel", "triangular_kernel", "cosine_kernel", "exponential_kernel"])

    bpe = BPE(merges=num_merges)

    tokens = bpe.encode(text)

    train_tokens = tokens[:int(len(tokens) * 0.9)]

    model = MarkovChain(n=n, smoothing=smoothing, alpha=alpha, kernel=kernels[kernel])
    model.train(train_tokens)

    gen_tokens = model.generate(max_length=500)

    score = count_score(model, tokens, gen_tokens, train_tokens)

    return score


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение метрики:", study.best_value)
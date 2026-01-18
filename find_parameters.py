from bpe import BPE
from utils import *
from markov_chain import *

import optuna


corpus = load_corpus('data/war_and_peace.txt')
text = preprocess_text(corpus)

train_tokens = text[:int(len(text) * 0.4)]
test_tokens = text[int(len(text) * 0.4):int(len(text) * 0.6)]

text_train = " ".join(train_tokens)

kernels = {
    "gaussian_kernel": gaussian_kernel,
    "triangular_kernel": triangular_kernel,
    "cosine_kernel": cosine_kernel,
    "exponential_kernel": exponential_kernel,
    "no": no_kernel
}


def objective(trial):
    global text_train, kernels

    n = trial.suggest_int("n", 1, 7)
    num_merges = trial.suggest_int("num_merges", 1, 100)
    smoothing = trial.suggest_categorical("smoothing", ["none", "add-k", "interpolation"])
    alpha = 0.0
    if smoothing == "add-k":
        deg = trial.suggest_float("alpha_deg", 1, 10)
        alpha = 10 ** (-deg)

    kernel = "no"
    bandwidth = 1.0
    if smoothing == "interpolation":
        kernel = trial.suggest_categorical("kernel", ["gaussian_kernel", "triangular_kernel", "cosine_kernel", "exponential_kernel"])
        max_bandwidth = 2.0 * math.sqrt(n + 1)
        bandwidth = trial.suggest_float(
            "kernel_bandwidth",
            0.1,
            max_bandwidth,
            log=True
        )

    bpe = BPE(merges=num_merges)

    tokens = bpe.encode(text_train)

    train_tokens_enc = tokens[:int(len(tokens) * 0.7)]
    test_tokens_enc = tokens[int(len(tokens) * 0.7):]

    model = MarkovChain(n=n, smoothing=smoothing, alpha=alpha, kernel=kernels[kernel],kernel_bandwidth=bandwidth)
    model.train(train_tokens_enc)

    gen_tokens = model.generate(max_length=500)
    text_int = [int(i) for i in gen_tokens]
    gen = bpe.decode(text_int)
    score = count_score(model, test_tokens_enc, gen.split(), train_tokens, test_tokens)

    return score


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение метрики:", study.best_value)

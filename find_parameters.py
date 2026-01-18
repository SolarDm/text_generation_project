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

    trial_num = trial.number
    print(f"\n{'=' * 60}")
    print(f"TRIAL #{trial_num} начат")
    print('=' * 60)

    n = trial.suggest_int("n", 1, 7)
    num_merges = trial.suggest_int("num_merges", 0, 100)
    print(f"num merges = {num_merges}")
    smoothing = trial.suggest_categorical("smoothing", ["none", "add-k", "interpolation"])
    alpha = 0.0
    if smoothing == "add-k":
        alpha = trial.suggest_float("alpha", 1e-5, 1.0, log=True)

    kernel = "no"
    if smoothing == "interpolation":
        kernel = trial.suggest_categorical("kernel", ["gaussian_kernel", "triangular_kernel", "cosine_kernel", "exponential_kernel"])

    bpe = BPE(merges=num_merges)

    tokens = bpe.encode(text_train)

    train_tokens_enc = tokens[:int(len(tokens) * 0.7)]
    test_tokens_enc = tokens[int(len(tokens) * 0.7):]

    model = MarkovChain(n=n, smoothing=smoothing, alpha=alpha, kernel=kernels[kernel])
    model.train(train_tokens_enc)

    gen_tokens = model.generate(max_length=500)
    text_int = [int(i) for i in gen_tokens]
    gen = bpe.decode(text_int)

    score = count_score(model, test_tokens_enc, gen.split(), train_tokens, test_tokens)

    sample = gen[:50] + "..." if len(gen) > 50 else gen
    trial.set_user_attr("sample", sample)

    print(f"Score: {score:.4f}")
    print(f"Пример текста: '{sample}'")
    print(f"TRIAL #{trial_num} завершён")

    return score


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение метрики:", study.best_value)

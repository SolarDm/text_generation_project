import matplotlib.pyplot as plt

from bpe import BPE
from markov_chain import *
from utils import *

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

orders = [i for i in range(1, 10)]

def make_objective(smoothing, n):
    def objective(trial): 
        global text_train, kernels

        # num_merges = tsrial.suggest_int("num_merges", 1, 25)
        num_merges = 0
        alpha = 0.0
        if smoothing == "k-add":
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

        train_tokens_enc = tokens[:int(len(tokens) * 0.9)]
        test_tokens_enc = tokens[int(len(tokens) * 0.9):]

        model = MarkovChain(n=n, smoothing=smoothing, alpha=alpha, kernel=kernels[kernel],kernel_bandwidth=bandwidth)
        model.train(train_tokens_enc)

        gen_tokens = model.generate(max_length=500)
        text_int = [int(i) for i in gen_tokens]
        gen = bpe.decode(text_int)
        score = count_score(model, test_tokens_enc, gen.split(), train_tokens, test_tokens)

        return score if score > 1e-5 else -(num_merges / 100.0 + n / 12.0)

    return objective

with open("assets/file.txt", "w") as file:
    for sm in ['none', 'interpolation', 'k-add']:
        scores = list()

        for n in orders:
            file.write(f"sm={sm}, n={n}\n")
            file.write("-"*40 + "\n")
            study = optuna.create_study(study_name=f'Study for smoothing: {sm} and n: {n}', direction="maximize")

            study.optimize(make_objective(sm, n), n_trials=20)

            score = study.best_value
            file.write(str(score) + "\n")
            file.write(str(study.best_params))
            file.write("\n" + "-"*40 + "\n")
            scores.append(score)
        
        file.write(" ".join(list(map(str, scores))) + "\n")
        
        plt.plot(orders, scores, label=sm)

plt.xlabel("Длина n-граммы")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

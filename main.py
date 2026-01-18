from bpe import BPE
from markov_chain import MarkovChain, exponential_kernel, triangular_kernel, gaussian_kernel, cosine_kernel
from utils import load_corpus, preprocess_text


def main():
    corpus = load_corpus('data/war_and_peace.txt')
    tokens = preprocess_text(corpus)
    train_tokens = tokens[:int(len(tokens) * 0.1)]

    print(f"Всего токенов: {len(tokens)}")
    print(f"Токенов для обучения: {len(train_tokens)}")

    bpe = BPE(merges=10)

    text = " ".join(train_tokens)

    tokens = bpe.encode(text)

    orders = [1, 2, 3, 4, 5, 6, 7]
    generated_texts = []

    for n in orders:
        for sm in ['interpolation', 'k-add', 'none']:
            print(f"\nОбучение {n}-граммной модели со сглаживаеим {sm}...")
            model = MarkovChain(n=n, smoothing=sm, alpha=0.00001, kernel=exponential_kernel)
            model.train(tokens)

            text = model.generate(max_length=500)
            generated_texts.append(text)

            print(f"Сгенерированный текст (n={n}):")
            text = [int(i) for i in text]
            print(bpe.decode(text))
            print()


if __name__ == "__main__":
    main()

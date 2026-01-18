from bpe import BPE
from markov_chain import *
from utils import *

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

    for n in orders:
        for sm in ['k-add', 'interpolation']:
            print(f"\nОбучение {n}-граммной модели со сглаживаеим {sm}...")
            model = MarkovChain(n=n, smoothing=sm, alpha=0.01, kernel=exponential_kernel)
            model.train(tokens)

            # print(f'Качество модели: {calculate_perplexity(model, tokens)}')

            text = model.generate(max_length=500)

            print(f"Сгенерированный текст (n={n}):")
            text = [int(i) for i in text]
            gen = bpe.decode(text)
            print(gen)

            # print(f'Схожесть с изначальным тектом: {calculate_self_overlap(bpe.decode(text).split(), train_tokens, 2)}')
            print(f'Score: {count_score(model, tokens, gen.split(), train_tokens)}')

            print()


if __name__ == "__main__":
    main()

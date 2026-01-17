from bpe import BPE
from markov_chain import MarkovChain
from utils import load_corpus, preprocess_text

def main():
    print("Загрузка корпуса...")
    corpus = load_corpus('data/war_and_peace.txt')
    tokens = preprocess_text(corpus)
    
    train_tokens = tokens[:int(len(tokens) * 0.02)]
    
    print(f"Всего токенов: {len(tokens)}")
    print(f"Токенов для обучения: {len(train_tokens)}")

    bpe = BPE(merges=400)

    text = " ".join(train_tokens)

    tokens = bpe.encode(text)

    orders = [2, 3, 4, 5]
    generated_texts = []

    for n in orders:
        for sm in ['k-add']:
            print(f"\nОбучение {n}-граммной модели со сглаживаеим {sm}...")
            model = MarkovChain(n=n, smoothing=sm, alpha=0.0001)
            model.train(tokens)
        
            text = model.generate(max_length=1000)
            generated_texts.append(text)
            
            print(f"Сгенерированный текст (n={n}):")
            text = [int(i) for i in text]
            print(bpe.decode(text))
            print()

if __name__ == "__main__":
    main()
    
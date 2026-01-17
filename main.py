from markov_chain import MarkovChain
from utils import load_corpus, preprocess_text

def main():
    print("Загрузка корпуса...")
    corpus = load_corpus('data/war_and_peace.txt')
    tokens = preprocess_text(corpus)
    
    train_tokens = tokens[:int(len(tokens) * 0.01)]
    test_tokens = tokens[int(len(tokens) * 0.01):]
    
    print(f"Всего токенов: {len(tokens)}")
    print(f"Токенов для обучения: {len(train_tokens)}")
    print(f"Токенов для тестирования: {len(test_tokens)}")

    orders = [2, 3, 5, 10]
    generated_texts = []

    for n in orders:
        for sm in ['add-k', 'none']:
            print(f"\nОбучение {n}-граммной модели со сглаживаеим {sm}...")
            model = MarkovChain(n=n, smoothing=sm, alpha=0.5)
            model.train(train_tokens)

            # for context, _ in list(model.model.items()):
            #     print(context)
            #     print(model.model[context])
        
            text = model.generate(max_length=20)
            generated_texts.append(text)
            
            print(f"Сгенерированный текст (n={n}):")
            print(" ".join(text))
            print()

if __name__ == "__main__":
    main()
    
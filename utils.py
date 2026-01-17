import re
from typing import List

def load_corpus(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_text(text: str, lower_case: bool = True) -> List[str]:
    if lower_case:
        text = text.lower()

    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    tokens = text.split()
    
    return tokens

def split_train_test(tokens: List[str], test_size: float = 0.1) -> tuple:
    split_idx = int(len(tokens) * (1 - test_size))
    return tokens[:split_idx], tokens[split_idx:]

def calculate_perplexity(model, test_tokens: List[str]) -> float:
    log_prob_sum = 0
    total_transitions = 0

    for i in range(len(test_tokens) - model.n + 1):
        prefix = tuple(test_tokens[i:i + model.n - 1])
        next_word = test_tokens[i + model.n - 1]

        if prefix in model.model:
            counter = model.model[prefix]
            total = sum(counter.values())
            
            if model.smoothing == 'none':
                prob = counter[next_word] / total if total > 0 else 0
            else:
                prob = np.exp(counter.get(next_word, np.log(model.alpha / (total + model.alpha * len(model.vocab)))))
            
            if prob > 0:
                log_prob_sum += np.log(prob)
                total_transitions += 1

    if total_transitions == 0:
        return float('inf')
    
    avg_log_prob = log_prob_sum / total_transitions
    return np.exp(-avg_log_prob)
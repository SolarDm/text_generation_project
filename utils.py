import re
from typing import List
from markov_chain import MarkovChain
import math


def load_corpus(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_text(text: str, lower_case: bool = True) -> List[str]:
    if lower_case:
        text = text.lower()

    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'[iusv*\\()0123456789;?!":<>,h-]', ' ', text)
    text = text.replace('ё', 'е')
    text = text.replace('ъ', 'ь')
    tokens = text.split()
    
    return tokens


def split_train_test(tokens: List[str], test_size: float = 0.1) -> tuple:
    split_idx = int(len(tokens) * (1 - test_size))
    return tokens[:split_idx], tokens[split_idx:]


def calculate_perplexity(model: MarkovChain, test_tokens: List[str]) -> float:
    total_log_prob = 0.0
    N = len(test_tokens) - model.n

    if N <= 0:
        return float('inf')

    for i in range(len(test_tokens) - model.n):
        context = tuple(test_tokens[i:i + model.n])
        next_token = test_tokens[i + model.n]
        prob = model.get_probability(context, next_token)

        total_log_prob += math.log2(prob if prob > 0 else 1e-10)

    avg_log_prob = total_log_prob / N
    perplexity = 2 ** (-avg_log_prob)
    return perplexity


def calculate_self_overlap(generated_tokens: List[str], train_tokens: List[str], ngram_size: int = 5) -> float:
    train_ngrams = set()
    for i in range(len(train_tokens) - ngram_size + 1):
        train_ngrams.add(tuple(train_tokens[i:i + ngram_size]))

    generated_ngrams = []
    for i in range(len(generated_tokens) - ngram_size + 1):
        generated_ngrams.append(tuple(generated_tokens[i:i + ngram_size]))

    if not generated_ngrams:
        return 0.0

    overlap_count = 0
    for ngram in generated_ngrams:
        if ngram in train_ngrams:
            overlap_count += 1

    return overlap_count / len(generated_ngrams)


def count_score(model, tokens, gen, train, best_overlap_percent=0.2):
    perplexity = calculate_perplexity(model, tokens)
    overlap_2 = calculate_self_overlap(gen, train, 2)
    overlap_3 = calculate_self_overlap(gen, train, 3)
    avg_overlap = (overlap_2 + overlap_3) / 2

    ppl_term = min(1.0, 1.0 / (1.0 + perplexity / 3.0))

    if avg_overlap < best_overlap_percent:
        overlap_den = (best_overlap_percent - avg_overlap) * 20
    elif avg_overlap == best_overlap_percent:
        overlap_den = 1.0
    else:
        overlap_den = (avg_overlap - best_overlap_percent) * 5

    overlap_term = min(1.0 / overlap_den, 1.0)

    return ppl_term * overlap_term

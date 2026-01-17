import random
from collections import defaultdict, Counter
from typing import List, Tuple, Optional
import numpy as np


class MarkovChain:
    def __init__(
        self,
        n: int = 2,
        smoothing: str = 'none',
        alpha: float = 1.0,
        lambdas: Optional[List[float]] = None
    ):
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        self.lambdas = lambdas

        if smoothing == 'none':
            self.alpha = 0.0
        
        self.model = defaultdict(Counter)
        self.start_contexts = []
        self.vocab = set()
        self.vocab_size = 0
        
        self.lower_n_models = {} if smoothing == 'interpolation' else None
        
    
    def train(self, tokens: List[str]):
        self.training_tokens = tokens.copy()
        
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)
        
        if self.smoothing == 'interpolation':
            self._train_interpolation_models(tokens)
        else:
            self._collect_start_contexts(tokens)
            self._build_transition_matrix(tokens)
        
        self._count_probs()
        
    
    def _collect_start_contexts(self, tokens: List[str]):
        self.start_contexts = []

        first_context = tuple(tokens[:self.n])
        self.start_contexts.append(first_context)

        sentence_ends = {'.', '!', '?', ';', ':', '\n'}
        
        for i in range(len(tokens) - self.n):
            if tokens[i] in sentence_ends:
                context = tuple(tokens[i+1:i+1+self.n])
                if len(context) == self.n:
                    self.start_contexts.append(context)
    

    def _build_transition_matrix(self, tokens: List[str]):
        total_transitions = len(tokens) - self.n

        for i in range(total_transitions):
            context = tuple(tokens[i:i + self.n])
            next_token = tokens[i + self.n]
            
            self.model[context][next_token] += 1
    

    def _train_interpolation_models(self, tokens: List[str]):
        if self.lambdas is None:
            self.lambdas = [0.5 ** (self.n - i + 1) for i in range(self.n + 1)]
            total = sum(self.lambdas)
            self.lambdas = [l / total for l in self.lambdas]
        
        self.lower_n_models = {}
        
        for k in range(self.n + 1):
        
            model_k = defaultdict(Counter)
            
            if k == 0:
                token_counts = Counter(tokens)
                model_k[()] = token_counts
            else:
                for i in range(len(tokens) - k):
                    context = tuple(tokens[i:i + k])
                    next_token = tokens[i + k]
                    model_k[context][next_token] += 1
            
            self.lower_n_models[k] = model_k
        
        self.model = self.lower_n_models[self.n]
        self._collect_start_contexts(tokens)
   
    
    def _count_probs(self):
        if self.smoothing == 'interpolation':
            return
        
        self._apply_add_k_smoothing()


    def _apply_add_k_smoothing(self):
        for context, counter in list(self.model.items()):
            total_original = sum(counter.values())
            total_smoothed = total_original + self.alpha * len(self.vocab)
            
            smoothed_probs = {}
            for token, count in counter.items():
                smoothed_probs[token] = (count + self.alpha) / total_smoothed

            for token in self.vocab:
                if not (token in smoothed_probs):
                    smoothed_probs[token] = (self.alpha) / total_smoothed

            self.model[context] = smoothed_probs
    

    def get_probability(self, context: Tuple[str, ...], token: str) -> float:        
        if self.smoothing == 'interpolation':
            return self._get_interpolated_probability(context, token)
        else:
            return self._get_probability_for_context(context, token)
    

    def _get_probability_for_context(self, context: Tuple[str, ...], token: str) -> float:
        if context not in self.model:
            return 0.0
        
        counter = self.model[context]
        
        return counter.get(token, 0.0)
    

    def _get_interpolated_probability(self, context: Tuple[str, ...], token: str) -> float:
        total_prob = 0.0
        
        for k in range(self.n + 1):
            lambda_k = self.lambdas[k]
            
            if k == 0:
                unigram_model = self.lower_n_models[0]
                unigram_counter = unigram_model[()]
                prob_k = unigram_counter.get(token, 0) / self.vocab_size

            else:
                if len(context) >= k - 1:
                    sub_context = context[-(k - 1):] if k > 1 else ()
                    model_k = self.lower_n_models[k]
                    
                    if sub_context in model_k:
                        counter = model_k[sub_context]
                        total = sum(counter.values())
                        if total > 0:
                            prob_k = counter.get(token, 0) / total
                        else:
                            prob_k = 0.0
                    else:
                        prob_k = 0.0
                else:
                    prob_k = 0.0
            
            total_prob += lambda_k * prob_k
        
        return total_prob
    

    def generate(
        self,
        max_length: int = 100,
        temperature: float = 1.0,
        seed: Optional[str] = None
    ) -> List[str]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        context = list(random.choice(self.start_contexts))
        
        result = context.copy()
        
        for _ in range(max_length - len(context)):
            current_context = tuple(result[-self.n:]) if self.n > 0 else ()
            
            next_token = self._sample_next_token(current_context, temperature)
            
            if next_token is None:
                break
            
            result.append(next_token)
        
        return result
    
    def _sample_next_token(self, context: Tuple[str, ...], temperature: float) -> Optional[str]:
        candidates = []
        probabilities = []
        
        for token in self.vocab:
                prob = self.get_probability(context, token)
                candidates.append(token)
                probabilities.append(prob)

        
        probabilities = np.array(probabilities, dtype=float)

        if probabilities.sum() == 0:
            return random.choices(candidates, k=1)[0]

        if temperature != 1.0:
            probabilities = np.power(probabilities, 1.0 / temperature)
            probabilities = probabilities / probabilities.sum()
        
        return random.choices(candidates, weights=probabilities, k=1)[0]

from collections import Counter
from typing import List


class BPE:
    def __init__(self, merges: int = 300, code_whitespaces: bool = True):
        self.merges = merges
        self.vocab = {}
        self.merge_rules = []
        self.id_to_token = {}
        self.code_whitespaces = code_whitespaces

    def encode(self, text: str):
        tokens = list(text)

        whitespace_code = -1
        for char in set(tokens):
            if char not in self.vocab:
                idx = len(self.vocab)
                if char == ' ':
                    whitespace_code = idx
                self.vocab[char] = idx
                self.id_to_token[idx] = char

        token_ids = [self.vocab[char] for char in tokens]

        for it in range(self.merges):
            pair_counts = Counter()
            for i in range(len(token_ids) - 1):
                if not self.code_whitespaces and (token_ids[i] == whitespace_code or token_ids[i + 1] == whitespace_code):
                    continue
                pair = (token_ids[i], token_ids[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            mx = max(pair_counts.items(), key=lambda x: x[1])

            if mx[1] == 1:
                break

            best_pair = mx[0]

            new_token = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            new_id = len(self.vocab)

            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merge_rules.append(best_pair)

            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if (i < len(token_ids) - 1 and
                        token_ids[i] == best_pair[0] and
                        token_ids[i + 1] == best_pair[1]):
                    new_token_ids.append(new_id)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
            token_ids = new_token_ids

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        return ''.join(self.id_to_token[idx] for idx in token_ids)

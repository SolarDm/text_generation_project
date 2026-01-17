import random


class MarkovModel:
    def add_to_model(self, n, seq):
        seq = seq + [None]
        for i in range(len(seq)-n):
            gram = tuple(seq[i:i+n])
            next_item = seq[i+n]            
            if gram not in self.model:
                self.model[gram] = []
            self.model[gram].append(next_item)


    def __init__(self, n, seq):
        self.model = {}
        self.n = n
        self.add_to_model(n, seq)
    

    def gen_from_model(self, start=None, max_gen=100):
        if start is None:
            start = random.choice(list(self.model.keys()))
        output = list(start)
        for i in range(max_gen):
            start = tuple(output[-self.n:])
            next_item = random.choice(self.model[start])
            if next_item is None:
                break
            else:
                output.append(next_item)
        return output

    

# def markov_model(n, seq):
#     model = {}
#     add_to_model(model, n, seq)
    # return model

# def markov_model_from_sequences(n, sequences):
#     model = {}
#     for item in sequences:
#         add_to_model(model, n, item)
#     return model

# def markov_generate_from_sequences(n, sequences, count, max_gen=100):
#     starts = [item[:n] for item in sequences if len(item) >= n]
#     model = markov_model_from_sequences(n, sequences)
#     return [gen_from_model(n, model, random.choice(starts), max_gen)
#            for i in range(count)]

# def markov_generate_from_lines_in_file(n, filehandle, count, level='char', max_gen=100):
#     if level == 'char':
#         glue = ''
#         sequences = [item.strip() for item in filehandle.readlines()]
#     elif level == 'word':
#         glue = ' '
#         sequences = [item.strip().split() for item in filehandle.readlines()]
#     generated = markov_generate_from_sequences(n, sequences, count, max_gen)
#     return [glue.join(item) for item in generated]

# if __name__ == '__main__':
#     import sys
#     try:
#         n = int(sys.argv[1])
#     except (ValueError, IndexError):
#         n = 3
#     for item in markov_generate_from_lines_in_file(n, sys.stdin, 20):
#         print(item)
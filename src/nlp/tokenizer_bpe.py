from collections import Counter


def get_stats(tokens):
    pairs = Counter()
    for token in tokens:
        for i in range(len(token) - 1):
            pairs[(token[i], token[i + 1])] += 1
    return pairs


def train_bpe(corpus_tokens, num_merges=10):
    merges = []
    vocab = [list(t) for t in corpus_tokens]
    for _ in range(num_merges):
        stats = get_stats(vocab)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        merges.append(pair)
        new_vocab = []
        for token in vocab:
            i = 0
            merged = []
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                    merged.append(token[i] + token[i + 1])
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            new_vocab.append(merged)
        vocab = new_vocab
    return merges

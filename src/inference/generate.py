def greedy_decode(next_token_fn, prompt_tokens, max_new_tokens=32):
    tokens = list(prompt_tokens)
    for _ in range(max_new_tokens):
        tokens.append(next_token_fn(tokens))
    return tokens

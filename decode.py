import torch

import config


def align(S, song, level='word'):
    # finds monotonic path maximizing the cumulative similarity score
    # level = ['token', 'word']

    num_tokens, num_frames = S.shape

    DP = torch.zeros_like(S)
    DP[0, 0] = S[0, 0]

    for i in range(num_tokens):
        for j in range(1, num_frames):
            if i == 0:
                DP[i, j] = DP[i, j - 1]
            else:
                DP[i, j] = max(DP[i, j - 1], DP[i - 1, j - 1])
            DP[i, j] += S[i, j]

    token_alignment = [(0, 0) for _ in range(num_tokens)]
    start, end = num_frames  # end non-inclusive
    for token in reversed(1, range(num_tokens)):
        while DP[token, start - 1] > DP[token - 1, start - 1]:
            start -= 1
        token_alignment[token] = (start, end)
        end = start
    
    if level == 'token':
        return token_alignment
    

    words = song['words'] if config.use_chars else song['phonemes']
    word_alignment = [(0, 0) for _ in range(len(words))]
    l = r = 0
    for word in words:
        num_tokens = len(word['text'])
        r += num_tokens
        start = token_alignment[l][0]
        end = token_alignment[r - 1][1]
        word_alignment[word] = (start, end)
        l = r

    return word_alignment
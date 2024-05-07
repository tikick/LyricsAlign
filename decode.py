import torch

import config


def _align(S, song, level='word'):
    # finds monotonic path maximizing the cumulative similarity score
    # level = ['token', 'word']

    # take silence padding into account
    # alignment in spec frames, need to convert to seconds

    # assert level in ['token', 'word']

    num_tokens, num_frames = S.shape

    DP = torch.zeros_like(S)  # -np.inf?
    DP[0, 0] = S[0, 0]

    for i in range(num_tokens):
        for j in range(1, num_frames):
            if i == 0:
                DP[i, j] = DP[i, j - 1]
            else:
                DP[i, j] = max(DP[i, j - 1], DP[i - 1, j - 1])
            DP[i, j] += S[i, j]

    token_alignment = []
    start, end = num_frames  # end non-inclusive
    for token in reversed(1, range(num_tokens)):
        if token > 0:
            while DP[token, start - 1] > DP[token - 1, start - 1]:  # start should always be > 0, as token > 0
                start -= 1
        else:  # token == 0:
            start = 0
        token_alignment.append((start, end))
        end = start
        start -= 1

    token_alignment = list(reversed(token_alignment))
    
    if level == 'token':
        return token_alignment
    
    words = song['words'] if config.use_chars else song['phowords']
    word_alignment = []
    first_token = last_token = 1  # first and last token of current word, skip first silence token
    for i, word in enumerate(words):
        num_tokens = len(word)
        last_token = first_token + num_tokens - 1
        start = token_alignment[first_token][0]
        end = token_alignment[last_token][1]
        word_alignment.append((start, end))
        first_token = last_token + 2  # +1 space between words

    return word_alignment


def align(S, song, masked, level='word'):
    if masked:
        token_alignment = _align(S, song, level='token')
        mask = compute_line_mask(S, song, token_alignment)
        S = S * mask
    alignment = _align(S, song, level)
    return convert_frames_to_seconds(alignment)
    

def compute_line_mask(S, song, token_alignment):
    token_duration = 9 if config.use_chars else 17  # duration in frames (duration in seconds times fps)
    tol_window_length = 108

    mask = torch.zeros_like(S)

    lines = song['lines'] if config.use_chars else song['pholines']
    first_token = last_token = 1  # first token of current line, skip first silence token
    for i, line in enumerate(lines):
        num_tokens = len(line)
        last_token = first_token + num_tokens - 1
        middle_token = first_token + num_tokens / 2
        line_center = token_alignment[middle_token][0]
        line_start = line_center - (num_tokens - 1) * token_duration / 2
        line_end = line_center + (num_tokens + 1) * token_duration / 2 + 1 # +1 to make non-inclusive (?)

        mask[first_token:last_token, line_start:line_end] = 1
        # add linear tolerance window
        mask[first_token:last_token, line_start - tol_window_length:line_start] = torch.linspace(0, 1, tol_window_length)
        mask[first_token:last_token, line_end + 1:line_end + tol_window_length + 1] = torch.linspace(1, 0, tol_window_length)

        first_token += num_tokens + 1  # +1 space between lines
    
    return mask


def convert_frames_to_seconds(alignment):
    # convert (start, end) from spec frames to seconds
    fps = 43.07
    return [(start / fps, end / fps) for (start, end) in alignment]

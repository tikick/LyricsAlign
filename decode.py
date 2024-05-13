import numpy as np

import config


def _align(S, song, level='word'):
    # finds monotonic path maximizing the cumulative similarity score

    # take pre- and post-silence into account

    assert level in ['token', 'word']

    num_tokens, num_frames = S.shape

    DP = -np.inf * np.ones_like(S)
    parent = - np.ones_like(S, dtype=int)

    for i in range(num_tokens):
        for j in range(i, num_frames):
            if i == 0 and j == 0:
                DP[i, j] = S[i, j]
                parent[i, j] = -1
            elif i == 0:
                DP[i, j] = DP[i, j - 1] + S[i, j]
                parent[i, j] = i
            else:
                m = max(DP[i, j - 1], DP[i - 1, j - 1])
                DP[i, j] = m + S[i, j]
                parent[i, j] = i if m == DP[i, j - 1] else i - 1
    
    #print('DP =\n', DP)
    #print('parent =\n', parent)

    token_alignment = []
    token_start = token_end = num_frames
    for token in reversed(range(num_tokens)):
        assert token_start > 0
        token_start -= 1
        while parent[token, token_start] == token:
            token_start -= 1
        token_alignment.append((token_start, token_end))
        token_end = token_start

    token_alignment = list(reversed(token_alignment))
    
    if level == 'token':
        return token_alignment
    
    words = song['words'] if config.use_chars else song['phowords']
    word_alignment = []
    first_word_token = last_word_token = 1
    for word in words:
        num_word_tokens = len(word)
        last_word_token = first_word_token + num_word_tokens - 1
        word_start = token_alignment[first_word_token][0]
        word_end = token_alignment[last_word_token][1]
        word_alignment.append((word_start, word_end))
        first_word_token = last_word_token + 2  # +1 space between words

    return word_alignment


def align(S, song, level='word'):
    if config.masked:
        token_alignment = _align(S, song, level='token')
        mask = compute_line_mask(S, song, token_alignment)
        S = S * mask
    alignment = _align(S, song, level)
    return convert_frames_to_seconds(alignment)
    

def compute_line_mask(S, song, token_alignment):
    # take pre- and post-silence into account
    
    token_duration = 9 if config.use_chars else 17  # duration in frames (0.2 * fps and 0.4 * fps)
    tol_window_length = 108  # 2.5 * fps

    mask = np.zeros_like(S)
    num_tokens, num_frames = S.shape

    lines = song['lines'] if config.use_chars else song['pholines']
    first_line_token = last_line_token = 1
    for line in lines:
        num_line_tokens = len(line)
        last_line_token = first_line_token + num_line_tokens - 1
        middle_token = first_line_token + num_line_tokens // 2
        line_center = token_alignment[middle_token][0]
        line_start = max(line_center - (num_line_tokens - 1) * token_duration // 2, 0)
        line_end = min(line_center + (num_line_tokens + 1) * token_duration // 2 + 1, num_frames)  # +1 to make non-inclusive

        mask[first_line_token:last_line_token, line_start:line_end] = 1
        # add linear tolerance window
        # left tolerance window
        window_start = max(line_start - tol_window_length, 0)
        window_end = line_start
        mask[first_line_token:last_line_token, window_start:window_end] = \
            np.linspace(0, 1, tol_window_length)[tol_window_length - (window_end - window_start):]
        # right tolerance window
        window_start = line_end
        window_end = min(line_end + tol_window_length, num_frames)
        mask[first_line_token:last_line_token, window_start:window_end] = \
            np.linspace(1, 0, tol_window_length)[:window_end - window_start]

        first_line_token = last_line_token + 2  # +1 space between lines
    
    return mask


def convert_frames_to_seconds(alignment):
    # convert (start, end) from spec frames to seconds
    fps = 43.07
    return [(start / fps, end / fps) for (start, end) in alignment]


if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)

    #signal = np.random.rand(20)
    #signal = 2 * signal - 1
    signal = np.array([-1, 1, -1, 1, 1, 1, -1, -1, -1, -1])#[-0.88,  0.99, -0.66,  0.52,  0.15,  0.80, -0.60, -0.68, -0.70, -0.95])#, -0.84425061, -0.24026376, 0.90270039,  0.48420148,  0.68169934, -0.34080505,  0.98538769,  0.33862212, 0.61028969,  0.12509671])
    print(signal)

    #half_signal = np.concatenate((signal[:10:2], signal[10:20], signal[20::2]))
    half_signal = signal[::2].copy()
    print(half_signal)

    signal = np.expand_dims(signal, axis=0)
    half_signal = np.expand_dims(half_signal, axis=1)

    S = np.matmul(half_signal, signal)
    S = 0.5 * (S + 1)

    #aabbbacccc
    S = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 0, 0], # a
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0], # b
                  [1, 1, 0, 0, 0, 1, 0, 0, 0, 0], # a
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]) # c

    print('S =\n', S)

    alignment = _align(S, None, level='token')
    print('alignment =\n', alignment)
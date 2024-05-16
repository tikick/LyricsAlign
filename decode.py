import numpy as np
import wandb

import config


def _align(S, song, level='word'):
    # finds monotonic path maximizing the cumulative similarity score

    # NOTE: take pre- and post-silence into account

    assert np.all((S >= 0) & (S <= 1))

    assert level in ['token', 'word']

    wandb_images = []
    wandb_images.append(wandb.Image(S, caption='S'))

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
    
    wandb_images.append(wandb.Image(DP, caption='DP'))

    parent_image = np.zeros_like(parent)
    for i in range(num_tokens):
        for j in range(num_frames):
            if parent[i, j] == i:
                parent_image[i, j] = 1
    wandb_images.append(wandb.Image(parent_image, caption='parent'))

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

    alignment_image = np.zeros_like(DP)
    for token, time in enumerate(token_alignment):
        alignment_image[token, time[0]:time[1]] = 1
    wandb_images.append(wandb.Image(alignment_image, caption='token_alignment'))
    
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
    
    assert len(word_alignment) == len(song['gt_alignment'])

    alignment_image = np.zeros(shape=(len(words), S.shape[1]))
    for word, time in enumerate(word_alignment):
        alignment_image[word, time[0]:time[1]] = 1
    wandb_images.append(wandb.Image(alignment_image, caption='word_alignment'))

    alignment_image = np.zeros(shape=(len(words), len(song['gt_alignment'])))
    for word, time in enumerate(song['gt_alignment']):
        alignment_image[word, time[0]:time[1]] = 1
    wandb_images.append(wandb.Image(alignment_image, caption='gt_alignment'))

    wandb.log({song['id']: wandb_images})

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
    pass
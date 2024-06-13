import numpy as np
import wandb
import matplotlib.pyplot as plt

import config


def vertical_align(S, song, level, log, epoch):
    # finds monotonic path maximizing the cumulative similarity score, without horizontal score accumulation

    assert np.all((S >= 0) & (S <= 1))
    assert level in ['token', 'word']

    num_tokens, num_frames = S.shape

    DP = np.zeros_like(S)
    parent = np.empty_like(S, dtype=bool)  # True = left, False = up
    for i in range(num_tokens):
        for j in range(i, num_frames):
            if i == 0 and j == 0:
                DP[i, j] = S[i, j]
                parent[i, j] = False
            elif j == 0:
                DP[i, j] = DP[i - 1, j] + S[i, j]
                parent[i, j] = False
            elif i == 0:
                DP[i, j] = max(DP[i, j - 1], S[i, j])
                parent[i, j] = (DP[i, j] == DP[i, j - 1])
            else:
                DP[i, j] = max(DP[i, j - 1], DP[i - 1, j] + S[i, j])
                parent[i, j] = (DP[i, j] == DP[i, j - 1])
    
    token_alignment = []
    token_start = token_end = num_frames - 1  # token_end inclusive
    for token in reversed(range(num_tokens)):
        assert token_start > 0
        while parent[token, token_start]:  # while parent is left
            token_start -= 1
        token_alignment.append((token_start, token_end))
        token_end = token_start

    token_alignment = list(reversed(token_alignment))

    if log:
        token_alignment_image = np.zeros_like(S)
        for token, frames in enumerate(token_alignment):
            token_alignment_image[token, frames[0]:frames[1]] = 1
    
    if level == 'token':
        return token_alignment
    
    words = song['words'] if config.use_chars else song['phowords']
    word_alignment = []
    first_word_token = last_word_token = 0
    for word in words:
        num_word_tokens = len(word)
        last_word_token = first_word_token + num_word_tokens - 1
        word_start = token_alignment[first_word_token][0]
        word_end = token_alignment[last_word_token][1]
        word_alignment.append((word_start, word_end))
        first_word_token = last_word_token + 2  # +1 space between words
    
    assert len(word_alignment) == len(song['times'])

    if log:
        words = song['words']
        word_alignment_image = np.zeros(shape=(len(words), num_frames))
        gt_word_alignment_image = np.zeros(shape=(len(words), num_frames))
        for word, frames in enumerate(word_alignment):
            word_alignment_image[word, frames[0]:frames[1]] = 1
        fps = 43.07  # the number of spectrogram frames in a second
        for word, time in enumerate(song['times']):
            frames = (int(time[0] * fps), int(time[1] * fps))
            gt_word_alignment_image[word, frames[0]:frames[1]] = 1

    # log plots
    if log:
        tokens = chars(song['words']) if config.use_chars else phonemes(song['phowords'])    
        heights = [len(tokens)] * 3 + [len(song['words'])] * 2
        fig, axs = plt.subplots(5, 1, height_ratios=heights, 
                                figsize=(min(num_frames // 14, 100), min((sum(heights) + 20 * len(heights)) // 12, 100)))
        
        show(DP, axs[0], 'DP', tokens)
        show(S, axs[1], 'S', tokens)
        alignment_cmap = 'Blues'
        show(token_alignment_image, axs[2], 'token alignment', tokens, alignment_cmap)
        show(word_alignment_image, axs[3], 'word alignment', song['words'], alignment_cmap)
        show(gt_word_alignment_image, axs[4], 'ground truth word alignment', song['words'], alignment_cmap)

        fig.tight_layout()

        wandb.log({'media/' + song['id']: plt, 'media/epoch': epoch})
        #plt.show()
        plt.close()

    return word_alignment


def _align(S, song, level, log, epoch):
    # finds monotonic path maximizing the cumulative similarity score

    # NOTE: take pre- and post-silence into account  # REMOVED SILENCE PADDING IN data.py

    assert np.all((S >= 0) & (S <= 1))
    assert level in ['token', 'word']

    num_tokens, num_frames = S.shape

    DP = np.zeros_like(S)  # -np.inf * np.ones_like(S)
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
                parent[i, j] = i - 1 if m == DP[i - 1, j - 1] else i
    
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

    if log:
        token_alignment_image = np.zeros_like(DP)
        for token, frames in enumerate(token_alignment):
            token_alignment_image[token, frames[0]:frames[1]] = 1
    
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
    
    assert len(word_alignment) == len(song['times'])

    if log:
        words = song['words']
        word_alignment_image = np.zeros(shape=(len(words), num_frames))
        gt_word_alignment_image = np.zeros(shape=(len(words), num_frames))
        for word, frames in enumerate(word_alignment):
            word_alignment_image[word, frames[0]:frames[1]] = 1
        fps = 43.07  # the number of spectrogram frames in a second
        for word, time in enumerate(song['times']):
            frames = (int(time[0] * fps), int(time[1] * fps))
            gt_word_alignment_image[word, frames[0]:frames[1]] = 1

    # log plots
    if log:
        tokens = chars(song['words'], space_padding=1) if config.use_chars else phonemes(song['phowords'], space_padding=1)    
        heights = [len(tokens)] * 3 + [len(song['words'])] * 2
        fig, axs = plt.subplots(5, 1, height_ratios=heights, 
                                figsize=(min(num_frames // 14, 100), min((sum(heights) + 20 * len(heights)) // 12, 100)))
        
        show(S, axs[0], 'S', tokens)
        show(DP, axs[1], 'DP', tokens)
        alignment_cmap = 'Blues'
        show(token_alignment_image, axs[2], 'token alignment', tokens, alignment_cmap)
        show(word_alignment_image, axs[3], 'word alignment', song['words'], alignment_cmap)
        show(gt_word_alignment_image, axs[4], 'ground truth word alignment', song['words'], alignment_cmap)

        fig.tight_layout()

        wandb.log({'media/' + song['id']: plt, 'media/epoch': epoch})
        #plt.show()
        plt.close()

    return word_alignment


def show(data, ax, title, ytick_labels, cmap='hot'):
    im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='none')
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_yticks(ticks=np.arange(data.shape[0]), labels=ytick_labels)
    ax.tick_params(axis='both', labelsize=6)


def align(S, song, level, log, epoch):
    if config.masked:
        token_alignment = vertical_align(S, song, level='token', log=False, epoch=-1)
        mask = compute_line_mask(S, song, token_alignment)
        S = S * mask
    alignment = vertical_align(S, song, level, log, epoch)  # was: _align
    return convert_frames_to_seconds(alignment)
    

def compute_line_mask(S, song, token_alignment):
    # all spaces get 0 !!!
    
    token_duration = 9 if config.use_chars else 17  # duration in frames (0.2 * fps and 0.4 * fps)
    tol_window_length = 108  # 2.5 * fps

    mask = np.zeros_like(S)
    num_tokens, num_frames = S.shape

    lines = song['lines'] if config.use_chars else song['pholines']
    first_line_token = past_last_line_token = 0
    for line in lines:
        num_line_tokens = len(line)
        past_last_line_token = first_line_token + num_line_tokens
        middle_token = first_line_token + num_line_tokens // 2
        line_center = token_alignment[middle_token][0]
        line_start = max(line_center - (num_line_tokens - 1) * token_duration // 2, 0)
        line_end = min(line_center + (num_line_tokens + 1) * token_duration // 2 + 1, num_frames)  # +1 to make non-inclusive

        mask[first_line_token:past_last_line_token, line_start:line_end] = 1
        # add linear tolerance window
        # left tolerance window
        window_start = max(line_start - tol_window_length, 0)
        window_end = line_start
        mask[first_line_token:past_last_line_token, window_start:window_end] = \
            np.linspace(0, 1, tol_window_length)[tol_window_length - (window_end - window_start):]
        # right tolerance window
        window_start = line_end
        window_end = min(line_end + tol_window_length, num_frames)
        mask[first_line_token:past_last_line_token, window_start:window_end] = \
            np.linspace(1, 0, tol_window_length)[:window_end - window_start]

        first_line_token = past_last_line_token + 1  # +1 space between lines
    
    return mask


def convert_frames_to_seconds(alignment):
    # convert (start, end) from spec frames to seconds
    fps = 43.07
    return [(start / fps, end / fps) for (start, end) in alignment]


# get yticklabels for plots
def chars(words):
    lyrics = ' '.join(words)
    return [c for c in lyrics]
def phonemes(phowords):
    phonemes = []
    for phoword in phowords:
        phonemes += phoword + [' ']
    phonemes = phonemes[:-1]
    return phonemes


if __name__ == '__main__':
    pass
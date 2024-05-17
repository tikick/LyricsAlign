import numpy as np
import wandb
import matplotlib.pyplot as plt

import config


def encode_words(words, space_padding):
    lyrics = ' '.join(words)
    lyrics = ' ' * space_padding + lyrics + ' ' * space_padding
    
    chars = []
    for c in lyrics:
        chars.append(c)
    return chars

def encode_phowords(phowords, space_padding):
    phonemes = []
    for phoword in phowords:
        phonemes += phoword + [' ']
    phonemes = phonemes[:-1]
    phonemes = [' '] * space_padding + phonemes + [' '] * space_padding

    phonemes = []
    for p in phonemes:
        phonemes.append(p)
    return phonemes

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def _align(S, song, level='word'):
    # finds monotonic path maximizing the cumulative similarity score

    # NOTE: take pre- and post-silence into account

    assert np.all((S >= 0) & (S <= 1))
    assert level in ['token', 'word']

    fps = 43.07  # the number of spectrogram frames in a second

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
    
    # DP = DP / np.max(DP)

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

    token_alignment_image = np.zeros_like(DP)
    for token, frames in enumerate(token_alignment):
        token_alignment_image[token, frames[0]:frames[1]] = 1
    #wandb_images.append(wandb.Image(alignment_image, caption='token_alignment'))
    
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

    words = song['words']  # only for logging
    word_alignment_image = np.zeros(shape=(len(words), num_frames))
    for word, frames in enumerate(word_alignment):
        word_alignment_image[word, frames[0]:frames[1]] = -1
    for word, time in enumerate(song['gt_alignment']):
        frames = (time[0] * fps, time[1] * fps)
        word_alignment_image[word, frames[0]:frames[1]] = 1
    #wandb_images.append(wandb.Image(alignment_image, caption='gt_alignment'))

    #wandb.log({song['id']: wandb_images})

    def log():
        if config.use_chars:
            tokens = encode_words(song['words'], space_padding=config.context + 1)
        else:
            tokens = encode_phowords(song['phowords'], space_padding=config.context + 1)
        time = [i for i in range(num_frames)]
        
        S, _ = heatmap(S, tokens, time)
        DP, _ = heatmap(DP, tokens, time)
        token_alignment_image, _ = heatmap(token_alignment_image, tokens, time)
        word_alignment_image, _ = heatmap(word_alignment_image, song['words'], time)
        wandb.log({'S': S,
                   'DP': DP,
                   'token_alignment': token_alignment_image,
                   'word_alignment': word_alignment_image})
        
    log()

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
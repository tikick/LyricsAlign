import os
from typing import Optional, List, Tuple

import numpy as np


class ShortAudioError(Exception):
    pass


def trellis_segmentation(
    symbol_probs: np.ndarray, resolution: float, blank_probs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    Predicts an alignment path given a symbol probability matrix indicating the probability
    for each symbol in the lyrics sequence at each point in the audio.
    :param symbol_probs: Emission probability matrix [Time, Lyrics Sequence]
    Each value has to belong to [0, +inf[ and a higher value means a lyric token at a given
    time position is more likely. Preferably, the sum of values across the lyric sequence
    is equal to 1 - blank_probs for a given time position. There must be at least one strictly
    positive value for the decoding to work. Tip: If your emission matrix is
    of shape [Time, Vocabulary], you convert it with new_emission = old_emission[:, lyrics]
    with "lyrics" the lyrics sequence.
    :param resolution: Time in second between 2 temporal indexes in the probability matrix
    :param blank_probs: Emission probability vector of no token [Time], set to None for similarity
    :return: The timing of each lyric token, their index, and their decoded score.
    """
    check_trellis_inputs(symbol_probs, resolution, blank_probs)

    num_frames = np.size(symbol_probs, 0)
    num_tokens = np.size(symbol_probs, 1)
    symbol_probs_log = np.log(symbol_probs)
    if blank_probs is None:
        blank_probs = np.ones(num_frames)
    blank_probs_log = np.log(blank_probs)

    # An additional token to represent the start of sentence.
    # An addition observation for convenience.
    trellis = -np.inf * np.ones((num_frames + 1, num_tokens + 1), dtype=np.float32)
    trellis[:, 0] = 0
    trellis_path = np.zeros(np.shape(trellis), dtype=int)

    # forward pass
    for t in range(num_frames):
        keep_state = trellis[t, 1:] + blank_probs_log[t]
        next_state = trellis[t, :-1] + symbol_probs_log[t, :]
        trellis_path[t + 1, 1:] = np.array(next_state > keep_state, dtype=int)
        trellis[t + 1, 1:] = np.maximum(keep_state, next_state)

    # backward pass
    index_token = np.size(trellis, -1) - 1
    t_last_token = np.argmax(trellis[:, index_token])
    symbol_index = []
    symbol_timings = []
    symbol_timings_prob = []
    for t in range(t_last_token, 0, -1):
        if trellis_path[t, index_token]:
            # the token just changed
            symbol_index.append(index_token - 1)
            symbol_timings.append((t - 1) * resolution)
            symbol_timings_prob.append(symbol_probs[t - 1, index_token - 1])
            index_token -= 1

    return np.array(symbol_timings[::-1]), symbol_index[::-1], np.array(symbol_timings_prob[::-1])


def check_trellis_inputs(
    symbol_probs: np.ndarray,
    resolution: float,
    blank_probs: Optional[np.ndarray] = None,
) -> None:
    """
    Check symbol probabilities/scores/resolution of the model are well-formed and can be used
    for alignment. If yes, do nothing. If no, raises an AssertionError exception (or
    ShortAudioError in case there are not enough timestamps)
    :param symbol_probs: See trellis_segmentation
    :param resolution: See trellis_segmentation
    :param blank_probs: See trellis_segmentation
    """
    print(type(symbol_probs))
    assert (
        symbol_probs.ndim == 2
    ), f"Need to provide [time, symbols] shape for symbol_probs not {symbol_probs.shape}"
    # Raise ShortAudioError in case we don't have enough timeframes in the symbol_probs
    if symbol_probs.shape[0] < symbol_probs.shape[1]:
        raise ShortAudioError(
            "There needs to be at least as much time frames than tokens to decode them left to"
            f" right, but here there are {symbol_probs.shape[0]} time frames and"
            f" {symbol_probs.shape[1]} tokens"
        )
    assert np.min(symbol_probs) >= 0, "Need to provide positive values in symbol_probs"
    assert np.max(symbol_probs) > 0, "Need to provide at least one value > 0 in symbol_probs"
    assert (
        blank_probs is None or np.min(blank_probs) >= 0
    ), "If blank_probs is given it must be positive"
    assert blank_probs is None or (
        blank_probs.shape[0] == symbol_probs.shape[0]
    ), "If blank_probs given, it must have the same number of timestamps as symbol_probs"
    assert resolution > 0, f"Invalid resolution: {resolution} - Needs to be positive"


def test_trellis_segmentation_real_data() -> None:
    # Test if we get an expected result on real data with custom decode scheme
    # See if it matches what we see here:
    # https://tutorials.pytorch.kr/intermediate/forced_alignment_with_torchaudio_tutorial.html
    labels = [
        "-",
        "|",
        "E",
        "T",
        "A",
        "O",
        "N",
        "I",
        "H",
        "S",
        "R",
        "D",
        "L",
        "U",
        "M",
        "W",
        "C",
        "F",
        "G",
        "Y",
        "P",
        "B",
        "V",
        "K",
        "'",
        "X",
        "J",
        "Q",
        "Z",
    ]
    prob_matrix_path = os.path.join(os.path.dirname(__file__), "symbol_probs.npy")
    symbol_probs = np.load(prob_matrix_path)
    transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
    dictionary = {c: i for i, c in enumerate(labels)}
    lyrics = np.array([dictionary[c] for c in transcript])
    blank_probs = symbol_probs[:, 0]
    symbol_probs = symbol_probs[:, lyrics]
    symbol_timings, symbol_index, decoded_symbol_probs = trellis_segmentation(
        symbol_probs,
        resolution=0.020118,
        blank_probs=blank_probs,
    )

    # correct shape
    assert symbol_timings.ndim == 1 and decoded_symbol_probs.ndim == 1
    # we are getting high probabilities when the model works well:
    assert np.mean(decoded_symbol_probs) > 0.9
    num_check = 0  # I : 0.604
    assert 0.55 < symbol_timings[num_check] < 0.65 and transcript[num_check] == "I"
    num_check = 2  # H : 0.724
    assert 0.70 < symbol_timings[num_check] < 0.75 and transcript[num_check] == "H"
    num_check = 6  # T : 0.885
    assert 0.85 < symbol_timings[num_check] < 0.90 and transcript[num_check] == "T"
    num_check = 21  # B : 1.891
    assert 1.87 < symbol_timings[num_check] < 1.92 and transcript[num_check] == "B"
    num_check = 31  # A : 2.515
    assert 2.49 < symbol_timings[num_check] < 2.54 and transcript[num_check] == "A"

    # my
    print(blank_probs.shape)
    print(symbol_probs.shape)
    print(symbol_timings.shape)
    print(decoded_symbol_probs.shape)


#test_trellis_segmentation_real_data()


import config
from utils import load

def align(S: np.ndarray, song):
    waveform = load(song['audio_path'], sr=config.sr)
    duration = len(waveform) / config.sr
    print(duration)
    fps = S.shape[1] / duration
    print(fps)
    token_starts, _, _ = trellis_segmentation(S.transpose(), resolution=1/fps)

    words = song['words'] if config.use_chars else song['phowords']
    word_alignment = []
    first_word_token = last_word_token = 0
    for word in words:
        num_word_tokens = len(word)
        last_word_token = first_word_token + num_word_tokens - 1
        word_start = token_starts[first_word_token]
        word_end = token_starts[last_word_token]
        word_alignment.append((word_start, word_end))
        first_word_token = last_word_token + 2  # +1 space between words
        
    return word_alignment

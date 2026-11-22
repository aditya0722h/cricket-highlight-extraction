"""Local-normalization peak selection; existing fusion_engine.py is untouched.

Fixed defaults chosen before human-label evaluation:
* centered rolling z-score: 120 seconds, clipped to [-3, 3]
* centered moving-average smoothing: 5 seconds
* equal fusion weights: one third per channel
* peak non-maximum-suppression separation: 30 seconds
* windows begin 12 seconds before a peak and end when the score falls below
  its rolling median or 20 seconds pass; final lengths are clamped to 8--30 s.
"""

import numpy as np

DEFAULT_ROLLING_SEC = 120
DEFAULT_SMOOTH_SEC = 5
MIN_PEAK_SEPARATION_SEC = 30
MIN_WINDOW_SEC = 8
MAX_WINDOW_SEC = 30
PRE_PEAK_SEC = 12
POST_PEAK_LIMIT_SEC = 20


def _centered_mean(values, width):
    """Centered mean with edge windows shortened instead of zero-padded."""
    values = np.asarray(values, dtype=float)
    width = max(1, int(width))
    left = width // 2
    right = width - left - 1
    padded = np.pad(values, (left, right), mode="constant")
    counts = np.convolve(np.ones(len(values)), np.ones(width), mode="same")
    return np.convolve(padded, np.ones(width) / width, mode="valid") * width / counts


def rolling_zscore(values, window_sec=DEFAULT_ROLLING_SEC):
    """Centered local z-score using sums; zero standard deviation maps to zero."""
    values = np.asarray(values, dtype=float)
    width = max(1, int(window_sec))
    left, right = width // 2, width - width // 2 - 1
    padded = np.pad(values, (left, right), mode="constant")
    valid = np.pad(np.ones(len(values)), (left, right), mode="constant")
    kernel = np.ones(width)
    sums = np.convolve(padded, kernel, mode="valid")
    squares = np.convolve(padded * padded, kernel, mode="valid")
    counts = np.convolve(valid, kernel, mode="valid")
    means = sums / counts
    variance = np.maximum(squares / counts - means * means, 0.0)
    std = np.sqrt(variance)
    z = np.divide(values - means, std, out=np.zeros_like(values), where=std > 0)
    return np.clip(z, -3.0, 3.0)


def rolling_median(values, window_sec=DEFAULT_ROLLING_SEC):
    """Simple centered median; the short match timelines make clarity preferable."""
    values = np.asarray(values, dtype=float)
    half = max(1, int(window_sec)) // 2
    return np.array([
        np.median(values[max(0, index - half):min(len(values), index + half + 1)])
        for index in range(len(values))
    ])


def prepare_scores(raw_scores, rolling_sec=DEFAULT_ROLLING_SEC,
                   smooth_sec=DEFAULT_SMOOTH_SEC):
    """Apply the predetermined local normalization and smoothing steps."""
    return _centered_mean(rolling_zscore(raw_scores, rolling_sec), smooth_sec)


def select_peak_windows(scores, rolling_sec=DEFAULT_ROLLING_SEC,
                        min_separation_sec=MIN_PEAK_SEPARATION_SEC,
                        duration_sec=None):
    """Rank local maxima and make variable windows around non-suppressed peaks."""
    scores = np.asarray(scores, dtype=float)
    if duration_sec is None:
        duration_sec = float(len(scores))
    local_median = rolling_median(scores, rolling_sec)
    selected = []
    # Boolean masking is equivalent to testing every accepted peak's distance,
    # but avoids repeatedly scanning the selected list for long matches.
    suppressed = np.zeros(len(scores), dtype=bool)
    for index in np.argsort(scores)[::-1]:
        if suppressed[index]:
            continue
        start = max(0.0, float(index - PRE_PEAK_SEC))
        end = min(duration_sec, float(index + POST_PEAK_LIMIT_SEC))
        for future in range(index + 1, min(len(scores), index + POST_PEAK_LIMIT_SEC + 1)):
            if scores[future] < local_median[future]:
                end = float(future)
                break
        end = max(end, start + MIN_WINDOW_SEC)
        end = min(end, start + MAX_WINDOW_SEC, duration_sec)
        if end > start:
            selected.append({
                "start": round(start, 2), "end": round(end, 2),
                "score": float(scores[index]), "peak_sec": int(index),
            })
            suppressed[max(0, index - min_separation_sec + 1):
                       min(len(scores), index + min_separation_sec)] = True
    return selected


def peak_windows(raw_scores, rolling_sec=DEFAULT_ROLLING_SEC,
                 smooth_sec=DEFAULT_SMOOTH_SEC, duration_sec=None):
    """Convenience wrapper returning ranked variable windows and smoothed scores."""
    smoothed = prepare_scores(raw_scores, rolling_sec, smooth_sec)
    return select_peak_windows(smoothed, rolling_sec, duration_sec=duration_sec), smoothed

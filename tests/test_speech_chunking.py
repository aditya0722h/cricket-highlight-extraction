import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor

from speech_channel.speech_channel_pipeline import BoundedFeatureExtractor


def test_bounded_feature_extraction_matches_original_log_mel():
    random = np.random.default_rng(42)
    waveform = random.normal(0.0, 0.1, size=48_321).astype(np.float32)

    expected = FeatureExtractor()(waveform)
    actual = BoundedFeatureExtractor(
        FeatureExtractor(),
        batch_frames=127,
    )(waveform)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

from types import SimpleNamespace

import numpy as np
import soundfile as sf

from whisper_transcriber.audio import load_audio, resample
from whisper_transcriber.model import transcribe_audio
from whisper_transcriber.segmenter import SAMPLE_RATE


def test_load_audio_stereo_44k_to_mono_16k(tmp_path):
    rate = 44_100
    t = np.arange(rate) / rate
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1)
    path = tmp_path / "a.wav"
    sf.write(path, stereo, rate)
    audio = load_audio(path)
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert abs(len(audio) - SAMPLE_RATE) <= 2


def test_resample_identity():
    x = np.ones(100, dtype=np.float32)
    assert resample(x, SAMPLE_RATE) is x


def test_transcribe_audio_joins_segments_with_fake_model():
    class FakeModel:
        def transcribe(self, audio, **kwargs):
            assert kwargs["language"] == "en"
            return [SimpleNamespace(text=" hello "), SimpleNamespace(text="world ")], None

    assert transcribe_audio(FakeModel(), np.zeros(16000, dtype=np.float32)) == "hello world"

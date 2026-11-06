import numpy as np

from whisper_transcriber.segmenter import SAMPLE_RATE, SpeechSegmenter, rms

CHUNK = SAMPLE_RATE * 30 // 1000


def tone(ms, amplitude=0.3, frequency=220):
    samples = SAMPLE_RATE * ms // 1000
    t = np.arange(samples) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def silence(ms):
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.float32)


def feed(segmenter, audio):
    output = []
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        output += segmenter.process(audio[i:i + CHUNK])
    return output


def test_rms_of_silence_and_tone():
    assert rms(silence(30)) == 0.0
    assert abs(rms(tone(30, amplitude=0.3)) - 0.3 / np.sqrt(2)) < 0.01


def test_one_utterance_between_silences():
    utterances = feed(SpeechSegmenter(), np.concatenate([silence(500), tone(1000), silence(1000)]))
    assert len(utterances) == 1
    assert 1.0 <= len(utterances[0]) / SAMPLE_RATE <= 1.6


def test_two_utterances_separated_by_long_pause():
    audio = np.concatenate([silence(300), tone(600), silence(1000), tone(600), silence(1000)])
    assert len(feed(SpeechSegmenter(), audio)) == 2


def test_short_pause_does_not_split_utterance():
    audio = np.concatenate([tone(600), silence(300), tone(600), silence(1000)])
    assert len(feed(SpeechSegmenter(), audio)) == 1


def test_brief_noise_is_discarded():
    assert feed(SpeechSegmenter(), np.concatenate([silence(300), tone(60), silence(1000)])) == []


def test_flush_closes_utterance_in_progress():
    segmenter = SpeechSegmenter()
    feed(segmenter, np.concatenate([silence(200), tone(800)]))
    assert segmenter.speaking
    assert len(segmenter.flush()) == 1


def test_maximum_duration_cut():
    assert len(feed(SpeechSegmenter(max_ms=1000), tone(3000))) >= 2

# Whisper Transcriber

Live microphone and audio file transcription in English, running on a CPU. It wraps OpenAI's [Whisper](https://github.com/openai/whisper) through [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) and adds a small voice activity detector that turns a continuous microphone stream into complete sentences.

## Specifications

| | |
|---|---|
| Model | Whisper, English-only checkpoints (`tiny.en`, `base.en`, `small.en`, `medium.en`) |
| Default | `base.en` for the microphone, `small.en` for files |
| Runtime | CTranslate2 through `faster-whisper`, int8 quantization |
| Hardware | CPU; no GPU required |
| Input | Microphone, or any file `soundfile` can read (wav, flac, ogg...); converted to 16 kHz mono |
| Segmentation | Energy-based VAD for the microphone; Silero VAD for files |

## Usage

Install the dependencies as described in the [main README](../README.md#quick-start). On Linux, microphone capture also needs PortAudio (`sudo apt install libportaudio2`). Models are downloaded on first use.

**Microphone**

```bash
python -m whisper_transcriber.transcribe_mic
python -m whisper_transcriber.transcribe_mic --model tiny.en --output transcript.txt
```

| Option | Default | Description |
|---|---|---|
| `--model` | `base.en` | Whisper checkpoint |
| `--threshold` | `0.005` | Minimum RMS energy that counts as speech. Lower it if speech is missed, raise it if noise is picked up |
| `--silence-ms` | `700` | Pause that ends a sentence |
| `--device` | system default | Microphone index, from `python -m sounddevice` |
| `--output` | none | File to append the transcript to |

**Audio file**

```bash
python -m whisper_transcriber.transcribe_file audio.wav
python -m whisper_transcriber.transcribe_file audio.wav --model base.en --output transcript.txt
```

## How it works

```
microphone → 30 ms chunks → voice activity detector → complete sentence → Whisper → text
```

**Quantization.** Neural network weights are normally 32-bit floats. Storing them as 8-bit integers makes the model about four times smaller and faster on a CPU, with a small loss of quality. This is what makes transcription practical without a GPU.

**Voice activity detection** ([`segmenter.py`](segmenter.py)). Whisper processes clips of up to 30 seconds, not endless streams, so the audio has to be cut somewhere, ideally between sentences:

1. The RMS energy of each 30 ms chunk is compared to a threshold to decide whether it contains speech.
2. A sentence begins at the first speech chunk, keeping the previous 300 ms so the first word is not clipped.
3. A sentence ends after 700 ms of continuous silence, and is then sent to Whisper.
4. Very short sounds such as clicks or coughs are discarded, and sentences are cut at 25 seconds.

Energy-based detection is simple and transparent, but loud background noise can confuse it. File transcription therefore uses the neural Silero VAD included in `faster-whisper`.

**Resampling** ([`audio.py`](audio.py)). Whisper expects 16 kHz mono. Other files are averaged to one channel and resampled with a polyphase anti-aliasing filter.

## Choosing a model

| Model | Parameters | Notes |
|---|---|---|
| `tiny.en` | 39 M | Fastest, lowest accuracy |
| `base.en` | 74 M | Default for the microphone: keeps up with live speech on a laptop CPU |
| `small.en` | 244 M | Default for files: more accurate, slower than real time on a laptop CPU |
| `medium.en` | 769 M | Highest accuracy offered here, noticeably slower |

## Evaluation

Measured on 100 clips (8.5 minutes) of LibriSpeech `dev-clean`, on an Intel Core i7-8550U with int8 quantization. References and transcripts were normalized the same way (lowercase, no punctuation or digits).

| Model | WER | CER | Compute time per second of audio |
|---|---|---|---|
| `base.en` | 3.8 % | 1.8 % | 0.6 s |
| `small.en` | 3.4 % | 1.4 % | 1.9 to 2.7 s |

This is a small sample of clean, read speech, so it serves as a sanity check rather than a benchmark. Expect higher error rates with background noise or casual speech. Compute time needs to stay below 1 s per second of audio for live use.

## Testing

[`tests/test_segmenter.py`](../tests/test_segmenter.py) and [`tests/test_audio_model.py`](../tests/test_audio_model.py) cover segmentation on synthetic signals (sentence detection, short pauses, noise bursts, maximum duration), resampling and output handling. Live capture depends on the audio hardware and has no automated test.

## Code layout

| File | Purpose |
|---|---|
| `segmenter.py` | Energy-based voice activity detector |
| `audio.py` | Audio loading and resampling |
| `model.py` | `faster-whisper` loading and transcription |
| `transcribe_mic.py` | Microphone command line tool |
| `transcribe_file.py` | Audio file command line tool |

## References

- Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356), 2022.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2).

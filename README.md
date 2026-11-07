# ASR_experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

English speech-to-text, approached from two directions: a ready-to-use transcriber built on a state-of-the-art pretrained model, and a complete speech recognition system written and trained from scratch.

| Project | Description | Runs on |
|---|---|---|
| [`whisper_transcriber`](whisper_transcriber/README.md) | Live microphone and audio file transcription with Whisper (`faster-whisper`), quantized for CPU | Any PC, no GPU required |
| [`asr_from_scratch`](asr_from_scratch/README.md) | Conformer-CTC acoustic model, n-gram beam search decoder and time-budgeted multi-GPU training pipeline | Trains on 2 x T4 GPUs (Kaggle); inference on CPU |

The projects are independent, and each README has the full specification. They meet in one place: `asr_from_scratch` can evaluate its model side by side with Whisper on the same audio.

## Results

| Model | WER | CER | Measured on |
|---|---|---|---|
| Conformer-CTC + 3-gram language model, 55 minutes of training | 17.1 % | 7.2 % | LibriSpeech `test-clean`, 2,620 clips |
| Whisper `small.en`, pretrained | 3.4 % | 1.4 % | LibriSpeech `dev-clean`, 100 clips |

These come from different clips, so they show the scale of the difference rather than an exact comparison.

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/carlos-vi/ASR_experiments.git
cd ASR_experiments
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Transcribe from the microphone with Whisper:

```bash
python -m whisper_transcriber.transcribe_mic
```

Transcribe a file with the custom model, using the [pretrained weights](asr_from_scratch/README.md#use-the-pretrained-model).

Check the custom model's full pipeline on synthetic data, in about a minute on a CPU:

```bash
python -m asr_from_scratch.train --synthetic --minutes 1 --d-model 64 --layers 2 \
    --batch-seconds 20 --lr 2e-3 --workers 0
```

To train the real model, follow the [Kaggle notebook](asr_from_scratch/notebooks/train_kaggle.ipynb).

## Repository layout

```
ASR_experiments/
├── whisper_transcriber/     Whisper-based transcriber
├── asr_from_scratch/        Conformer-CTC model and training pipeline
│   └── notebooks/           Kaggle training notebook
├── tests/                   Test suite
├── requirements.txt
└── LICENSE
```

## Tests

```bash
pytest
```

## License

The code in this repository is released under the [MIT License](LICENSE).

Third-party data and models keep their own licenses:

| Resource | Used for | License |
|---|---|---|
| [LibriSpeech](https://www.openslr.org/12/) | Training and evaluation | CC BY 4.0 |
| [LibriSpeech language models](https://www.openslr.org/11/) | Beam search decoding | Public domain |
| [Whisper](https://github.com/openai/whisper) | `whisper_transcriber` | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | `whisper_transcriber` | MIT |

None of these are stored in this repository: they are downloaded when needed. See each project's README for citations.

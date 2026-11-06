# ASR_experiments

English speech-to-text, approached from two directions: a ready-to-use transcriber built on a state-of-the-art pretrained model, and a complete speech recognition system written and trained from scratch.

| Project | Description | Runs on |
|---|---|---|
| [`whisper_transcriber`](whisper_transcriber/README.md) | Live microphone and audio file transcription with Whisper (`faster-whisper`), quantized for CPU | Any PC, no GPU required |
| [`asr_from_scratch`](asr_from_scratch/README.md) | Conformer-CTC acoustic model, n-gram beam search decoder and time-budgeted multi-GPU training pipeline | Trains on 2 x T4 GPUs (Kaggle); inference on CPU |

The projects are independent, and each README has the full specification. They meet in one place: `asr_from_scratch` can evaluate its model side by side with Whisper on the same audio.

## Quick start

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Transcribe from the microphone with Whisper:

```bash
python -m whisper_transcriber.transcribe_mic
```

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
└── requirements.txt
```

## Tests

```bash
pytest
```

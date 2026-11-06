# ASR from Scratch

A complete English speech recognition system built from first principles in PyTorch: audio features, a Conformer acoustic model, CTC training, and a beam search decoder with an n-gram language model. Everything except the tensor library and the audio file reader is implemented in this package.

The training pipeline is designed around one constraint: get the best possible model out of **about one hour on two NVIDIA T4 GPUs** (a free Kaggle session).

## Specifications

| | |
|---|---|
| Architecture | Conformer encoder with a CTC output layer |
| Size | 6.2 M parameters: `d_model` 144, 12 blocks, 4 attention heads, convolution kernel 31 |
| Input | 16 kHz mono audio, converted to 80-band log-Mel features (25 ms window, 10 ms hop) |
| Subsampling | Two strided convolutions, 40 ms per output frame |
| Output | 28 characters (`a-z`, space, apostrophe) plus the CTC blank |
| Dataset | LibriSpeech `train-clean-100` + `train-clean-360` (about 460 h) |
| Training | AdamW, mixed precision, DistributedDataParallel, time-based learning rate schedule |
| Augmentation | SpecAugment |
| Decoding | Greedy, or prefix beam search with a pretrained 3-gram language model |

## Results

**The model has not been trained on the full dataset yet.** WER and CER on LibriSpeech `test-clean` will be reported here once the first training run is complete.

As a reference for expectations, this is a compact character-level model with one hour of training, compared to the hundreds of thousands of hours of audio behind Whisper. The notebook measures both on the same clips. Performance will also be lower on microphone audio than on the audiobooks it trains on, since LibriSpeech contains clean, read speech.

### Verification so far

These checks confirm that the pipeline is correct. They do not predict accuracy on unseen speech.

| Check | Outcome |
|---|---|
| Memorizing 12 real LibriSpeech clips (`d_model` 144, 4 blocks, CPU) | CER 0.5 % after 100 steps, 0 % after 300 steps |
| Learning synthetic data with 2 distributed processes (CPU) | Validation CER 2.4 %; stopped at the 2.0-minute budget |
| Beam search with the real 3-gram model on simulated noisy output | Separates words correctly and rejects out-of-vocabulary merges |
| Log-Mel features compared with `torchaudio` | Relative difference of 1e-7 |
| LibriSpeech download and extraction, on a 15-minute subset of `dev-clean` | 158 clips with correct transcripts |

GPU training with mixed precision on the T4s and the download of the full training set will run for the first time in the notebook.

## Quick start

### Train on Kaggle

Open [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) on Kaggle, select **GPU T4 x2** and enable internet access, then run all cells. The notebook downloads the data, trains, evaluates on the test set and leaves `best.pt` in the output panel.

The training command it runs:

```bash
torchrun --nproc_per_node=2 -m asr_from_scratch.train \
    --train data/train.100/manifest.jsonl data/train.360/manifest.jsonl \
    --val data/validation/manifest.jsonl \
    --output checkpoints --minutes 55
```

| Option | Default | Description |
|---|---|---|
| `--minutes` | `55` | Training time budget |
| `--batch-seconds` | `160` | Seconds of audio per batch on each GPU |
| `--lr` | `1e-3` | Peak learning rate |
| `--d-model`, `--layers`, `--heads`, `--conv-kernel` | `144`, `12`, `4`, `31` | Model size |
| `--eval-minutes` | `10` | Time between validation runs and checkpoints |
| `--resume` | off | Continue from `last.pt` |

### Evaluate

```bash
# Greedy decoding
python -m asr_from_scratch.evaluate checkpoints/best.pt data/test/manifest.jsonl

# Beam search with the language model, tuning its weights on the validation set
python -m asr_from_scratch.evaluate checkpoints/best.pt data/test/manifest.jsonl \
    --lm 3-gram.pruned.3e-7.arpa.gz --tune-manifest data/validation/manifest.jsonl

# Compare with Whisper on the same clips
python -m asr_from_scratch.evaluate checkpoints/best.pt data/test/manifest.jsonl \
    --max-clips 100 --whisper small.en
```

The language model is the [3-gram model distributed with LibriSpeech](https://www.openslr.org/11/).

### Transcribe a file

```bash
python -m asr_from_scratch.inference checkpoints/best.pt audio.wav
python -m asr_from_scratch.inference checkpoints/best.pt audio.wav --lm 3-gram.pruned.3e-7.arpa.gz
```

## Designing for a one-hour budget

| Decision | Why |
|---|---|
| **Time-based schedule** | The run length is set in minutes, and the learning rate follows the clock instead of a step count. The warm-up and decay cycle always completes, and the run ends on schedule regardless of the speed of the GPU assigned. |
| **Two GPUs with DDP** | Each GPU runs its own process and computes gradients on different data, which nearly doubles the throughput with little overhead. |
| **Compact Conformer** | Convolution and attention capture local and global context, and the Conformer is an established architecture for ASR. A compact size leaves more weight updates in the budget, and small models converge faster. |
| **More data, not more epochs** | For the same number of updates, seeing 460 h of different audio generalizes better than repeating 100 h several times. |
| **Features on the GPU** | Spectrograms and SpecAugment run as GPU batch operations, so the few CPU cores available only have to read files. |
| **Duration-bucketed batches** | Batches hold a fixed amount of audio from clips of similar length, which minimizes the GPU time spent on padding. |
| **Language model at decoding** | A pretrained n-gram model adds knowledge of English that needs no training time at all. |

## How it works

### From sound to features

A microphone samples air pressure 16,000 times per second, producing a one-dimensional waveform. Two people saying the same word produce very different waveforms, so the signal needs a better representation.

Speech sounds differ in which frequencies carry energy. The audio is cut into overlapping 25 ms windows and the Fourier transform of each one gives its energy per frequency. These frequencies are then grouped into 80 Mel bands, which are narrow at low frequencies and wide at high ones, as in human hearing, and the logarithm compresses the dynamic range. The result is a matrix of 100 frames per second by 80 bands, normalized per utterance so that recording volume does not matter.

### CTC: learning without alignments

Training data consists of audio and text pairs, with no information about which frame corresponds to which letter. For every frame, the model outputs a probability distribution over the characters plus a special *blank* symbol. To read the output, repeated symbols are collapsed and then blanks are removed:

```
h h _ e _ l l _ l o o  →  hello
```

The blank separates true double letters (`l _ l` gives `ll`, while `l l` gives `l`). The CTC loss sums the probability of every alignment that collapses to the correct text, using dynamic programming, so the model discovers the alignment on its own.

### The Conformer

Two strided convolutions first reduce the sequence to 25 frames per second, which is still plenty for the 12 to 15 characters of typical speech, and makes attention much cheaper, as its cost grows with the square of the length. Each of the 12 blocks then applies:

```
half feed-forward → self-attention → convolution → half feed-forward
```

Self-attention lets each frame draw on the whole utterance, with relative positional encoding so that scores depend on the distance between frames. The depthwise convolution captures local patterns such as phoneme transitions. Convolution normalization uses LayerNorm rather than the original BatchNorm, which would be affected by padding and require synchronization across GPUs.

### Decoding with a language model

CTC decides each character almost independently, so greedy decoding can write words that sound right but do not exist. [`decoder.py`](decoder.py) implements a prefix beam search that keeps the most likely transcriptions at each frame, merging the alignments that collapse to the same text. Each time a word is completed, it adds the 3-gram language model score:

```
score = log P(audio | text) + alpha · log P(text) + beta · number of words
```

Hypotheses are also penalized as soon as the word being spelled can no longer become a known word. The weights `alpha` and `beta` are tuned on the validation set.

### Measuring quality

**WER** (Word Error Rate) is the number of word substitutions, deletions and insertions needed to turn the output into the reference, divided by the number of reference words. **CER** is the same measure on characters. Both are computed after the same text normalization: lowercase, without punctuation or digits.

## Limitations

- Trained on clean, read audiobooks, so noise, casual speech and strong accents will degrade accuracy.
- Output is lowercase, without punctuation or digits.
- Inference processes complete clips rather than streaming.

Natural next steps are noise augmentation, sub-word (BPE) tokens and longer training.

## Code layout

| File | Purpose |
|---|---|
| `vocabulary.py` | Character vocabulary and text normalization |
| `features.py` | Log-Mel spectrogram and SpecAugment |
| `model.py` | Conformer-CTC model |
| `decoder.py` | Greedy and prefix beam search decoding |
| `language_model.py` | ARPA n-gram language model |
| `data.py` | Dataset, manifests and duration-bucketed batching |
| `prepare_data.py` | LibriSpeech download and manifest creation |
| `train.py` | Time-budgeted distributed training |
| `evaluate.py` | WER and CER evaluation, language model tuning, Whisper comparison |
| `inference.py` | Audio file transcription |
| `metrics.py` | WER and CER |
| `synthetic.py` | Synthetic data for smoke tests |

## References

- Gulati et al., [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100), 2020.
- Graves et al., [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf), 2006.
- Hannun et al., [First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs](https://arxiv.org/abs/1408.2873), 2014.
- Park et al., [SpecAugment](https://arxiv.org/abs/1904.08779), 2019.
- Dai et al., [Transformer-XL](https://arxiv.org/abs/1901.02860), 2019 (relative positional encoding).
- Panayotov et al., [LibriSpeech: An ASR Corpus Based on Public Domain Audio Books](https://www.danielpovey.org/files/2015_icassp_librispeech.pdf), 2015.

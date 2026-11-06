import numpy as np
import pytest
import torch

from asr_from_scratch.data import DurationBatchSampler, SpeechDataset, collate
from asr_from_scratch.decoder import beam_search_decode, greedy_decode
from asr_from_scratch.features import LogMel, SpecAugment
from asr_from_scratch.language_model import UNKNOWN_LOG10, ArpaLanguageModel
from asr_from_scratch.metrics import cer, wer
from asr_from_scratch.model import ConformerCTC, count_parameters, subsampled_lengths
from asr_from_scratch.synthetic import generate
from asr_from_scratch.train import learning_rate, main as train_main
from asr_from_scratch.vocabulary import CHAR_TO_ID, VOCAB_SIZE, decode, encode, normalize

ARPA = """\\data\\
ngram 1=6
ngram 2=3

\\1-grams:
-1.0\t<unk>\t0.0
-1.0\t<s>\t-0.3
-1.0\t</s>\t0.0
-1.0\tTHE\t-0.3
-3.0\tCAT\t-0.3
-3.0\tCAP\t-0.3

\\2-grams:
-0.1\t<s> THE
-0.2\tTHE CAT
-4.0\tTHE CAP

\\end\\
"""


def small_model() -> ConformerCTC:
    return ConformerCTC(d_model=32, layers=2, heads=2, conv_kernel=7)


def posteriors(sequence: str) -> np.ndarray:
    """Near-deterministic log-probabilities; '_' stands for the blank."""
    ids = [0 if char == "_" else CHAR_TO_ID[char] for char in sequence]
    log_probs = np.full((len(ids), VOCAB_SIZE), -10.0, dtype=np.float32)
    log_probs[np.arange(len(ids)), ids] = 0.0
    return log_probs


def greedy(sequence: str) -> str:
    log_probs = torch.from_numpy(posteriors(sequence))[None]
    return greedy_decode(log_probs, torch.tensor([log_probs.size(1)]))[0]


def test_normalize_and_encode():
    assert normalize("Hello,   World! 42") == "hello world"
    assert normalize("Don’t 'quote' café-bar") == "don't quote cafe bar"
    assert decode(encode("It's fine")) == "it's fine"


def test_greedy_collapses_repeats_and_blanks():
    assert greedy("hh_e_ll_lo") == "hello"


def test_blank_between_equal_letters_keeps_double_letter():
    assert greedy("ll") == "l"
    assert greedy("l_l") == "ll"


@pytest.mark.parametrize("sequence", ["hh_e_ll_lo", "l_l", "the_ cc_a_t"])
def test_beam_search_without_lm_matches_greedy_on_clear_input(sequence):
    assert beam_search_decode(posteriors(sequence)) == greedy(sequence)


def test_language_model_backoff(tmp_path):
    path = tmp_path / "lm.arpa"
    path.write_text(ARPA)
    lm = ArpaLanguageModel(path)
    assert lm.order == 2
    assert lm.score(["<s>"], "the") == pytest.approx(-0.1 * np.log(10))
    assert lm.score(["cat"], "the") == pytest.approx((-0.3 - 1.0) * np.log(10))
    assert lm.score(["the"], "dog") == pytest.approx((-0.3 + UNKNOWN_LOG10) * np.log(10))


def test_language_model_resolves_acoustic_ambiguity(tmp_path):
    path = tmp_path / "lm.arpa"
    path.write_text(ARPA)
    log_probs = posteriors("the ca_x")
    # Acoustically, "p" is slightly preferred over "t".
    log_probs[-1, CHAR_TO_ID["x"]] = -10.0
    log_probs[-1, CHAR_TO_ID["p"]] = -0.5
    log_probs[-1, CHAR_TO_ID["t"]] = -1.0
    assert beam_search_decode(log_probs) == "the cap"
    assert beam_search_decode(log_probs, ArpaLanguageModel(path)) == "the cat"


def test_language_model_keeps_spaces_between_known_words(tmp_path):
    path = tmp_path / "lm.arpa"
    path.write_text(ARPA)
    # The space is acoustically weaker than the blank, yet "thecat" is not a word.
    log_probs = posteriors("the_cat")
    log_probs[3, CHAR_TO_ID[" "]] = -1.0
    log_probs[3, 0] = -0.5
    assert beam_search_decode(log_probs) == "thecat"
    assert beam_search_decode(log_probs, ArpaLanguageModel(path)) == "the cat"


def test_wer_cer():
    assert wer(["a b c d"], ["a b x d"]) == 0.25
    assert cer(["cats"], ["cuts"]) == 0.25
    assert wer(["a b"], ["a b"]) == 0.0


def test_logmel_ignores_padding():
    torch.manual_seed(0)
    features = LogMel()
    wave = torch.randn(8000)
    alone, frames = features(wave[None], torch.tensor([8000]))
    padded, _ = features(torch.cat([wave, torch.zeros(8000)])[None], torch.tensor([8000]))
    # Only the edge frames see the padding through the STFT window, which barely
    # shifts the per-utterance statistics.
    assert (alone[0, :48] - padded[0, :48]).abs().mean() < 0.05
    assert frames.item() == 51
    assert (padded[0, 51:] == 0).all()


def test_specaugment_masks_only_with_zeros():
    x = torch.ones(4, 200, 80)
    masked = SpecAugment()(x, torch.tensor([200, 150, 100, 50]))
    assert masked.shape == x.shape
    assert set(masked.unique().tolist()) <= {0.0, 1.0}
    assert masked.sum() < x.sum()


def test_model_shapes_and_lengths():
    model = small_model().eval()
    log_probs, lengths = model(torch.randn(2, 16000), torch.tensor([16000, 8000]))
    assert log_probs.shape[0] == 2 and log_probs.shape[2] == VOCAB_SIZE
    assert lengths[0] == subsampled_lengths(torch.tensor(101)) == log_probs.shape[1]
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(2, log_probs.shape[1]), atol=1e-4)


def test_padding_does_not_change_the_output():
    model = small_model().eval()
    wave = torch.randn(16000)
    with torch.no_grad():
        alone, lengths = model(wave[None], torch.tensor([16000]))
        padded, padded_lengths = model(torch.cat([wave, torch.zeros(16000)])[None], torch.tensor([16000]))
    assert lengths == padded_lengths
    assert torch.allclose(alone[0, :18], padded[0, :18], atol=1e-1)


def test_default_model_size_is_budget_friendly():
    assert 4e6 < count_parameters(ConformerCTC()) < 10e6


def test_learning_rate_schedule_reaches_peak_and_decays():
    assert learning_rate(0.0, 1.0) < 0.01
    assert learning_rate(0.1, 1.0) == pytest.approx(1.0)
    assert learning_rate(0.55, 1.0) == pytest.approx(0.5)
    assert learning_rate(1.0, 1.0) < 0.01


def test_batch_sampler_partitions_and_respects_duration_limit():
    durations = list(np.random.default_rng(0).uniform(1, 20, 500))
    ranks = [DurationBatchSampler(durations, 60, rank, 2) for rank in range(2)]
    batches = [list(sampler) for sampler in ranks]
    assert len(batches[0]) == len(batches[1]) > 0
    flat = [i for rank in batches for batch in rank for i in batch]
    assert len(flat) == len(set(flat))
    assert all(sum(durations[i] for i in batch) <= 60 for batch in batches[0])
    first = list(ranks[0])
    ranks[0].set_epoch(1)
    assert list(ranks[0]) != first


def test_minimal_synthetic_training_reduces_loss(tmp_path):
    manifest = generate(tmp_path, 32, seed=3)
    dataset = SpeechDataset(manifest)
    loader = torch.utils.data.DataLoader(dataset, 16, collate_fn=collate)
    torch.manual_seed(0)
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), 3e-3)
    ctc = torch.nn.CTCLoss(zero_infinity=True)
    losses = []
    for _ in range(6):
        total = 0.0
        for waves, lengths, targets, target_lengths, _texts in loader:
            log_probs, output_lengths = model(waves, lengths)
            loss = ctc(log_probs.transpose(0, 1), targets, output_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses.append(total)
    assert losses[-1] < losses[0]


def test_prepare_data_unpacks_parquet_shard(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow", exc_type=ImportError)
    parquet = pytest.importorskip("pyarrow.parquet", exc_type=ImportError)
    import io

    import soundfile as sf

    from asr_from_scratch import prepare_data

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16000, dtype=np.float32), 16000, format="FLAC")
    table = pa.table({
        "audio": [{"bytes": buffer.getvalue(), "path": "a.flac"}],
        "text": ["HELLO THERE"],
        "id": ["1-2-3"],
        "speaker_id": [1],
    })
    shard = tmp_path / "shard.parquet"
    parquet.write_table(table, shard)
    (tmp_path / "audio").mkdir()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(shard))

    budget = prepare_data._AudioBudget(0.0)
    rows = prepare_data._extract_shard("clean/test/0.parquet", tmp_path, budget)
    assert rows == [{"audio": "audio/1-2-3.flac", "text": "HELLO THERE", "duration": 1.0}]
    assert (tmp_path / "audio" / "1-2-3.flac").exists()
    assert not shard.exists()


def test_training_script_saves_and_resumes(tmp_path):
    arguments = ["--synthetic", "--output", str(tmp_path), "--d-model", "32", "--layers", "1",
                 "--heads", "2", "--conv-kernel", "7", "--batch-seconds", "8", "--workers", "0",
                 "--log-every", "1", "--val-clips", "4"]
    train_main(arguments + ["--max-steps", "2"])
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    train_main(arguments + ["--max-steps", "3", "--resume"])
    assert torch.load(tmp_path / "last.pt")["step"] == 3

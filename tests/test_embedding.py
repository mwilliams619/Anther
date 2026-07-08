"""Workstream E — MERT all-layer + multi-window embedding (model-free logic)."""
import numpy as np
import torch

from anther_ml.embedding import (
    plan_windows,
    aggregate_layers,
    embedding_config,
    get_embedding,
    SR,
    WINDOW_SECONDS,
    N_WINDOWS,
)


def test_plan_windows_evenly_spaced_full_track():
    n = int(60 * SR)  # 60s track
    windows = plan_windows(n, sr=SR, window_seconds=10.0, n_windows=3)
    assert len(windows) == 3
    win_len = int(10.0 * SR)
    for s, e in windows:
        assert e - s == win_len
        assert 0 <= s and e <= n           # never past the end
    assert windows[0][0] == 0
    assert windows[-1][1] == n             # last window ends at the track end


def test_plan_windows_short_clip_single_window():
    n = int(4 * SR)  # shorter than a 10s window
    windows = plan_windows(n, sr=SR, window_seconds=10.0, n_windows=3)
    assert windows == [(0, n)]


def test_plan_windows_deterministic():
    n = int(45 * SR)
    a = plan_windows(n)
    b = plan_windows(n)
    assert a == b  # corpus and query get identical plans


def test_aggregate_layers_mean_vs_last_differ():
    torch.manual_seed(0)
    # 5 hidden states, batch 2, time 7, hidden 4
    hs = [torch.randn(2, 7, 4) for _ in range(5)]
    mean_agg = aggregate_layers(hs, "mean")
    last_agg = aggregate_layers(hs, "last")
    assert mean_agg.shape == (2, 4)
    assert last_agg.shape == (2, 4)
    # 'last' equals time-mean of the final layer exactly.
    torch.testing.assert_close(last_agg, hs[-1].mean(dim=1))
    # all-layer mean genuinely differs from last-only.
    assert not torch.allclose(mean_agg, last_agg)


def test_aggregate_layers_is_mean_over_layers():
    hs = [torch.ones(1, 3, 2) * k for k in range(1, 6)]  # layers valued 1..5
    agg = aggregate_layers(hs, "mean")
    # time-mean of each layer = its constant; mean over layers = mean(1..5) = 3
    torch.testing.assert_close(agg, torch.full((1, 2), 3.0))


def test_embedding_config_records_settings():
    cfg = embedding_config(layer_aggregation="mean", normalize=True)
    assert cfg["phase"] == 2
    assert cfg["layer_aggregation"] == "mean"
    assert cfg["window_seconds"] == WINDOW_SECONDS
    assert cfg["n_windows"] == N_WINDOWS
    assert cfg["loudness_normalize"] is True
    assert cfg["sr"] == SR


class _FakeProcessor:
    """Mimics Wav2Vec2FeatureExtractor: turns a list of windows into a padded
    tensor batch. Records how many windows it saw."""
    def __init__(self):
        self.last_batch = None

    def __call__(self, windows, sampling_rate, return_tensors, padding):
        maxlen = max(len(w) for w in windows)
        arr = np.zeros((len(windows), maxlen), dtype=np.float32)
        for i, w in enumerate(windows):
            arr[i, : len(w)] = w
        self.last_batch = arr
        return {"input_values": torch.tensor(arr)}


class _FakeModel:
    """Returns deterministic per-window, per-layer hidden states so we can
    verify all-layer + multi-window pooling end to end without downloading MERT.
    """
    H = 4
    N_LAYERS = 5

    def __call__(self, input_values, output_hidden_states):
        b, t = input_values.shape
        # Each hidden state depends on layer index and the window's mean, so
        # aggregation over layers/windows is exercised non-trivially.
        wmean = input_values.mean(dim=1, keepdim=True)  # (b,1)
        hidden = []
        for layer in range(self.N_LAYERS):
            base = (wmean + layer).unsqueeze(1)  # (b,1,1)
            hidden.append(base.expand(b, 3, self.H).clone())

        class _Out:
            hidden_states = tuple(hidden)
        return _Out()


def test_embed_tracks_batched_matches_per_track():
    """Tier 1A invariant: batching several tracks into shared forward passes must
    yield each track's vector bit-for-bit identical to the one-at-a-time path."""
    from anther_ml.embedding import _embed_windows, embed_tracks_batched

    model, processor = _FakeModel(), _FakeProcessor()
    rng = np.random.default_rng(1)
    waveforms = [
        rng.standard_normal(int(45 * SR)).astype(np.float32),  # 3 full windows
        rng.standard_normal(int(60 * SR)).astype(np.float32),  # 3 full windows
        rng.standard_normal(int(4 * SR)).astype(np.float32),   # 1 short window
    ]

    reference = np.vstack([
        _embed_windows(
            model, processor, [y[s:e] for s, e in plan_windows(len(y))],
            device="cpu", layer_aggregation="mean",
        )
        for y in waveforms
    ])

    # batch_windows small enough to force multiple chunks within a length bucket.
    got = embed_tracks_batched(
        model, processor, waveforms, device="cpu",
        layer_aggregation="mean", batch_windows=4, use_fp16=False,
    )
    assert got.shape == reference.shape
    np.testing.assert_allclose(got, reference, rtol=1e-6, atol=1e-6)


def test_get_embedding_end_to_end_mocked(monkeypatch, tmp_path):
    import soundfile as sf
    import anther_ml.embedding as emb_mod

    # 45s of audio → 3 windows.
    y = np.random.default_rng(0).standard_normal(int(45 * SR)).astype(np.float32)
    path = tmp_path / "clip.wav"
    sf.write(path, y, SR)

    # Avoid loudness dependence in this logic test.
    monkeypatch.setattr(emb_mod, "loudness_normalize", lambda y, sr: y)

    vec = get_embedding(_FakeModel(), _FakeProcessor(), path, device="cpu",
                        layer_aggregation="mean", normalize=True)
    assert vec.shape == (_FakeModel.H,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()

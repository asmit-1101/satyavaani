"""EER, Platt calibration and voiceprints (sv_heads.py)."""
import numpy as np

from sv_heads import eer, platt, sigmoid, voiceprint, save_voiceprints, load_voiceprints


def test_eer_is_zero_when_classes_separate():
    score = np.r_[np.linspace(0, 1, 50), np.linspace(2, 3, 50)]
    label = np.r_[np.zeros(50), np.ones(50)]
    e, thr = eer(score, label)
    assert e == 0.0
    assert 1.0 <= thr <= 2.0


def test_eer_is_about_half_for_random_scores():
    rng = np.random.default_rng(0)
    e, _ = eer(rng.standard_normal(4000), rng.integers(0, 2, 4000))
    assert 0.45 < e < 0.55


def test_eer_matches_a_known_overlap():
    # 10% of each class sits on the wrong side of the gap -> EER 10%
    neg = np.r_[np.zeros(90), np.full(10, 5.0)]
    pos = np.r_[np.full(90, 10.0), np.full(10, 1.0)]
    e, _ = eer(np.r_[neg, pos], np.r_[np.zeros(100), np.ones(100)])
    assert abs(e - 0.10) < 1e-9


def test_eer_needs_both_classes():
    e, _ = eer([0.1, 0.2], [1, 1])
    assert np.isnan(e)


def _prob(a, b, z):
    return float(sigmoid(a * np.asarray(z) + b))


def test_platt_puts_50_percent_at_the_boundary():
    rng = np.random.default_rng(1)
    z = np.r_[rng.normal(-2, 1, 500), rng.normal(2, 1, 500)]
    y = np.r_[np.zeros(500), np.ones(500)]
    a, b = platt(z, y)
    assert a > 0
    assert abs(_prob(a, b, 0.0) - 0.5) < 0.1
    assert _prob(a, b, 3.0) > 0.9 and _prob(a, b, -3.0) < 0.1


def test_platt_stays_finite_on_very_confident_scores():
    # regression: the first version diverged here and the app reported "0% of clones caught"
    for scale in (1, 30, 300):
        z = np.r_[np.full(50, -scale), np.full(50, scale)].astype(float)
        y = np.r_[np.zeros(50), np.ones(50)]
        a, b = platt(z, y)
        assert np.isfinite(a) and np.isfinite(b)
        assert _prob(a, b, scale) > 0.5 > _prob(a, b, -scale)


def test_voiceprint_is_unit_length_and_averages():
    e = np.array([[1, 0, 0], [0, 1, 0]], "float32")
    vp = voiceprint(e)
    assert abs(np.linalg.norm(vp) - 1) < 1e-6
    assert np.allclose(vp, [2 ** -0.5, 2 ** -0.5, 0], atol=1e-6)


def test_voiceprints_round_trip(tmp_path):
    vps = {"u01": ("Asha", np.ones(192, "float32") / np.sqrt(192))}
    p = tmp_path / "voiceprints.npz"
    save_voiceprints(vps, p)
    back = load_voiceprints(p)
    assert list(back) == ["u01"]
    name, emb = back["u01"]
    assert name == "Asha" and emb.shape == (192,) and np.allclose(emb, vps["u01"][1])


def test_missing_voiceprint_file_means_no_voices(tmp_path):
    assert len(load_voiceprints(tmp_path / "none.npz")) == 0

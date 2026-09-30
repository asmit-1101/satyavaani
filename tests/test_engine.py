"""Live scoring engine (sv_engine.py) with stand-in models: the risk rule, the verdict bands and the
call scenarios from our findings report (section 5.2)."""
import numpy as np

from sv_engine import fuse, verdict_of, CONTEXT_PRIORS
from sv_heads import sigmoid
from helpers import SR, make_engine, speechlike, unit

MATCH = unit(np.r_[1.0, np.zeros(191)])                       # the enrolled caller
OTHER = unit(np.r_[0.3, np.sqrt(1 - 0.09), np.zeros(190)])    # another person, cosine 0.30
VPS = {"s01": ("Saksham", MATCH)}


def verdicts(engine, st):
    return [(r["t"], r["verdict"]) for r in st.history if r["scored"]]


def first_time(vs, verdict):
    return next((t for t, v in vs if v == verdict), None)


# ------------------------------------------------------------------ the rule itself
def test_verdict_bands():
    assert verdict_of(0) == verdict_of(44.9) == "PASS"
    assert verdict_of(45) == verdict_of(74.9) == "VERIFY"
    assert verdict_of(75) == verdict_of(100) == "HOLD"


def test_matching_voice_never_lowers_the_risk_of_a_fake():
    for p in (0.1, 0.5, 0.9, 0.99):
        with_match, _ = fuse(p, 1.0, 0.75, 0.0, have_spk=True)
        without, _ = fuse(p, 1.0, 0.75, 0.0, have_spk=False)
        assert with_match >= without - 1e-9


def test_fake_or_wrong_person_raises_the_risk():
    real_right, _ = fuse(0.05, 0.85, 0.75, 0.0, True)
    real_wrong, p_imp = fuse(0.05, 0.60, 0.75, 0.0, True)
    fake_right, _ = fuse(0.95, 0.85, 0.75, 0.0, True)
    assert sigmoid(real_right) < 0.45
    assert p_imp > 0.8 and sigmoid(real_wrong) > 0.75
    assert sigmoid(fake_right) > 0.75


def test_call_context_raises_the_risk():
    base, _ = fuse(0.3, 0.0, 0.75, 0.0, False)
    money, _ = fuse(0.3, 0.0, 0.75, CONTEXT_PRIORS["Asks for money / OTP"], False)
    assert money > base


# ------------------------------------------------------------------ whole calls
def test_genuine_caller_passes():
    eng = make_engine([0.05], [MATCH], VPS)
    st = eng.new_call("s01")
    eng.push(st, speechlike(21))
    vs = verdicts(eng, st)
    assert len(vs) == 7 and all(v == "PASS" for _, v in vs)
    assert "Voice matches Saksham" in st.history[-1]["reason"]


def test_ai_clone_is_held_at_the_first_window():
    eng = make_engine([0.95], [MATCH], VPS)          # the clone even matches the voiceprint
    st = eng.new_call("s01")
    eng.push(st, speechlike(9))
    vs = verdicts(eng, st)
    assert vs[0] == (3.0, "HOLD")
    assert st.alert.startswith("Likely AI-cloned voice")


def test_real_person_pretending_is_caught_after_4_s_of_speech():
    eng = make_engine([0.05], [OTHER], VPS)
    st = eng.new_call("s01")
    eng.push(st, speechlike(21))
    vs = verdicts(eng, st)
    assert vs[0][1] == "PASS"                        # identity isn't judged before 4 s of speech
    assert first_time(vs, "HOLD") is not None and first_time(vs, "HOLD") <= 12.0
    assert "does NOT match" in st.history[-1]["reason"]


def test_call_taken_over_by_a_clone_ends_in_hold():
    seq = [0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95]  # clone from the 3rd window
    eng = make_engine(seq, [MATCH], VPS)
    st = eng.new_call("s01")
    eng.push(st, speechlike(21))
    vs = verdicts(eng, st)
    assert vs[0][1] == vs[1][1] == "PASS"
    assert vs[-1][1] == "HOLD"


def test_one_odd_window_does_not_turn_a_real_call_red():
    seq = [0.05, 0.05, 0.05, 0.95, 0.05, 0.05, 0.05]  # one "shocked" window
    eng = make_engine(seq, [MATCH], VPS)
    st = eng.new_call("s01")
    eng.push(st, speechlike(21))
    assert all(v == "PASS" for _, v in verdicts(eng, st))


def test_silence_holds_the_score():
    eng = make_engine([0.95], [MATCH], VPS)
    st = eng.new_call("s01")
    eng.push(st, speechlike(6))
    risk_before = st.risk
    eng.push(st, np.zeros(12 * SR, "float32"))       # the caller goes quiet
    assert st.risk == risk_before and st.verdict == "HOLD"
    assert not st.history[-1]["scored"]


def test_audio_can_arrive_in_small_pieces():
    x = speechlike(12)
    a = make_engine([0.05], [MATCH], VPS); sa = a.new_call("s01"); a.push(sa, x)
    b = make_engine([0.05], [MATCH], VPS); sb = b.new_call("s01")
    for i in range(0, len(x), SR // 2):
        b.push(sb, x[i:i + SR // 2])
    assert [r["t"] for r in sa.history] == [r["t"] for r in sb.history]
    assert abs(sa.risk - sb.risk) < 1e-6


def test_unknown_claimed_caller_is_treated_as_unknown():
    eng = make_engine([0.05], [MATCH], VPS)
    st = eng.new_call("not-enrolled")
    assert st.claimed is None


def test_score_audio_api_shape():
    eng = make_engine([0.95], [MATCH], VPS)
    out = eng.score_audio(speechlike(6), claimed="s01")
    assert out["verdict"] == "HOLD" and out["claimed_name"] == "Saksham"
    assert out["deepfake_prob_mean"] > 0.9 and len(out["windows"]) == 2

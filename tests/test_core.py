"""Model-free unit tests: matcher, evaluation, config, dispatch argv."""
import numpy as np
import pytest

from silent_hermes import dispatch
from silent_hermes.evaluate import evaluate
from silent_hermes.matcher import Calibration, Matcher, dtw_batch
from silent_hermes.roi import interpolate, resample
from silent_hermes.store import NEGATIVE, Command, Settings, load_commands, load_settings

rng = np.random.default_rng(0)


def test_dtw_self_is_zero_and_warp_tolerant():
    a = rng.normal(size=(30, 16))
    stretched = np.repeat(a, 2, axis=0)  # same content, half speed
    other = rng.normal(size=(30, 16))
    d = dtw_batch(a, [a, stretched, other])
    assert d[0] == pytest.approx(0, abs=1e-5)
    assert d[1] < 0.05 < d[2]


def test_matcher_rejects_ambiguous_and_accepts_clear():
    m = Matcher(["a", "b"], Calibration(p_min=0.8, margin=0.3))
    assert m.decide(np.array([0.55, 0.45])).command is None
    r = m.decide(np.array([0.95, 0.05]))
    assert r.command == "a" and r.margin == pytest.approx(0.9)


def test_template_scores_pick_nearest_class():
    protos = {c: rng.normal(size=(20, 8)) for c in "abc"}
    temps = {c: [p + 0.05 * rng.normal(size=p.shape) for _ in range(4)] for c, p in protos.items()}
    m = Matcher(list("abc"))
    d = m.template_scores(protos["b"], temps)
    assert d.argmin() == 1


def _synthetic(n_cls=4, reps=6, noise=0.3, with_neg=True):
    protos = [rng.normal(size=(int(rng.integers(15, 30)), 12)) for _ in range(n_cls)]
    labels, embs = [], []
    ids = [f"c{i}" for i in range(n_cls)]
    for i, p in enumerate(protos):
        for _ in range(reps):
            labels.append(ids[i])
            embs.append(p + noise * rng.normal(size=p.shape))
    if with_neg:
        for _ in range(reps):
            labels.append(NEGATIVE)
            embs.append(rng.normal(size=(20, 12)))
        ids = ids + [NEGATIVE]
    scores = rng.normal(size=(len(labels), len(ids)))  # uninformative phrase channel
    return labels, embs, scores, ids


def test_evaluate_learns_to_trust_templates():
    labels, embs, scores, ids = _synthetic()
    rep = evaluate(labels, embs, scores, ids, max_false_accept=0.0)
    assert rep.top1_template_only == 1.0
    assert rep.accept_rate >= 0.9 and rep.false_accept_rate == 0.0


def test_interpolate_and_resample():
    lms = [None, np.zeros((4, 2)), None, np.ones((4, 2)), None]
    out = interpolate(lms)
    assert np.allclose(out[0], 0) and np.allclose(out[2], 0.5) and np.allclose(out[4], 1)
    assert len(resample(list(range(30)), 30, 25)) == 25


def test_shipped_configs_load():
    cmds = load_commands()
    assert len(cmds) >= 5 and all(c.phrase for c in cmds)
    s = load_settings()
    for c in cmds:  # every action builds a valid argv
        argv = dispatch.build_argv(c, s, text="hello" if c.action["type"] == "dictate" else None)
        assert argv[0] == s.hermes_bin


def test_dispatch_argv_shapes(tmp_path):
    s = Settings(tmp_path, tmp_path, tmp_path, tmp_path)
    assert dispatch.build_argv(Command("x", "p", {"type": "hermes", "prompt": "hi"}), s) == ["hermes", "-z", "hi"]
    assert dispatch.build_argv(Command("x", "p", {"type": "dictate", "prompt_template": "N: {text}"}), s, "abc")[-1] == "N: abc"
    with pytest.raises(ValueError):
        dispatch.build_argv(Command("x", "p", {"type": "shell"}), s)


def _words(text, t0=0.0, step=0.3):
    return [{"w": w, "s": round(t0 + i * step, 2), "e": round(t0 + i * step + 0.25, 2)} for i, w in enumerate(text.split())]


def test_find_phrases_locates_commands_and_fillers():
    from silent_hermes.script import find_phrases

    words = _words("um Hermes status report. The tea is getting cold on the table next GRE task okay "
                   "turn off the lights turn on the lights")
    targets = [("status", "hermes status report"), ("gre", "next gre task"), ("off", "turn off the lights"),
               ("on", "turn on the lights"), (NEGATIVE, "the tea is getting cold on the table")]
    hits = find_phrases(words, targets)
    assert [h.id for h in hits] == ["status", NEGATIVE, "gre", "off", "on"]
    assert hits[0].start == 0.3 and hits[0].text == "hermes status report"


def test_find_phrases_tolerates_one_slip_in_long_phrases_only():
    from silent_hermes.script import find_phrases

    long_ = find_phrases(_words("switch of the air conditioner"), [("ac", "switch off the air conditioner")])
    assert len(long_) == 1 and long_[0].errors == 1
    assert find_phrases(_words("turn of the lights"), [("off", "turn off the lights")])[0].errors == 1
    assert find_phrases(_words("next g task"), [("gre", "next gre task")]) == []  # 3 words: exact only


def test_make_script_has_reps_and_fillers():
    from silent_hermes.script import FILLERS, make_script

    cmds = load_commands()
    lines = make_script(cmds, reps=3)
    assert len(lines) == 3 * len(cmds) + len(FILLERS)
    assert sum(l.id == cmds[0].id for l in lines) == 3
    assert [l.n for l in lines] == list(range(1, len(lines) + 1))


def test_portal_api_basics(tmp_path):
    from fastapi.testclient import TestClient

    from silent_hermes.portal import app as portal_app

    class FakePortal:
        root = tmp_path
        def script(self):
            from silent_hermes.script import Line
            return [Line(1, "status", "hermes status report")]
        def _read(self, uid):
            return None

    c = TestClient(portal_app.create_app(FakePortal()))
    assert c.get("/api/script").json()[0]["text"] == "hermes status report"
    assert c.delete("/api/uploads/../../etc").status_code == 404
    assert c.delete("/api/uploads/20261018-000000-abcdef").status_code == 404

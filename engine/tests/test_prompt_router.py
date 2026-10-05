"""PromptRouter: text/voice → active moment classes, with intent inference."""
from models.prompt_router import PromptRouter
from models.clip_moment_detector import MOMENT_PROFILES


def _new_router():
    return PromptRouter(MOMENT_PROFILES)


def test_init_session_picks_classes_from_initial_prompt():
    r = _new_router()
    st = r.init_session("s1", "cake cutting and ring ceremony")
    assert "cake_cutting"  in st.active_classes
    assert "ring_ceremony" in st.active_classes
    assert "general_peak"  in st.active_classes  # always-on floor


def test_replace_intent_default():
    r = _new_router()
    r.init_session("s1", "cake cutting")
    t = r.update("s1", "first dance")
    assert t.intent == "replace"
    active = set(t.active)
    assert active == {"first_dance", "general_peak"}


def test_add_intent():
    r = _new_router()
    r.init_session("s1", "cake cutting")
    t = r.update("s1", "also catch confetti")
    assert t.intent == "add"
    assert "cake_cutting" in t.active
    assert "confetti_burst" in t.active


def test_remove_intent():
    r = _new_router()
    r.init_session("s1", "cake cutting and confetti")
    t = r.update("s1", "stop catching confetti")
    assert t.intent == "remove"
    assert "confetti_burst" not in t.active
    assert "cake_cutting" in t.active


def test_general_peak_is_floor():
    r = _new_router()
    r.init_session("s1", "cake cutting")
    t = r.update("s1", "stop catching cake cutting")
    # Even after removing all matches, general_peak survives.
    assert "general_peak" in t.active


def test_noop_when_unknown_words():
    r = _new_router()
    r.init_session("s1", "cake cutting")
    t = r.update("s1", "something unrelated")
    assert t.intent == "noop"


def test_history_capped():
    r = _new_router()
    r.init_session("s1", "cake")
    for i in range(60):
        r.update("s1", f"add confetti {i}")
    st = r.get("s1")
    assert len(st.history) <= 50

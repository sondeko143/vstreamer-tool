from vspeech.config import SubtitleTextConfig
from vspeech.worker import subtitle as subtitle_mod
from vspeech.worker.subtitle import Text
from vspeech.worker.subtitle import Texts
from vspeech.worker.subtitle import redraw_panel


class FakeCanvas:
    """Records the Tk calls redraw_panel makes, without needing a display."""

    def __init__(self, events: list):
        self.events = events

    def delete(self, tag: str):
        self.events.append(("delete", tag))

    def pack(self):
        self.events.append(("pack",))


def make_panel(anchor: str = "s", bb_width: int = 300, bb_height: int = 200) -> Texts:
    panel = Texts(
        tag="text",
        anchor=anchor,
        config=SubtitleTextConfig(anchor=anchor),
        bb_width=bb_width,
        bb_height=bb_height,
    )
    panel.values.append(Text(value="hello"))
    return panel


def install_recording_draw(monkeypatch, events: list) -> dict:
    """Replace draw_text_with_outline with a Tk-free recorder; return its kwargs."""
    captured: dict = {}

    def fake_draw(**kwargs):
        events.append(("draw",))
        captured.update(kwargs)

    monkeypatch.setattr(subtitle_mod, "draw_text_with_outline", fake_draw)
    return captured


def test_redraw_panel_clears_tag_before_drawing(monkeypatch):
    events: list = []
    install_recording_draw(monkeypatch, events)
    panel = make_panel()

    redraw_panel(FakeCanvas(events), panel)

    # Old text for this tag must be deleted before the new text is drawn,
    # otherwise successive frames stack on top of each other.
    assert events.index(("delete", "text")) < events.index(("draw",))


def test_redraw_panel_packs_after_drawing(monkeypatch):
    events: list = []
    install_recording_draw(monkeypatch, events)
    panel = make_panel()

    redraw_panel(FakeCanvas(events), panel)

    assert events.index(("draw",)) < events.index(("pack",))


def test_redraw_panel_draws_current_panel_state(monkeypatch):
    events: list = []
    captured = install_recording_draw(monkeypatch, events)
    panel = make_panel(anchor="s", bb_width=300)

    redraw_panel(FakeCanvas(events), panel)

    assert captured["texts"] == panel.texts
    assert captured["text_tag"] == panel.tag
    assert captured["anchor"] == panel.anchor
    assert captured["config"] is panel.config
    assert captured["text_coord_x"] == panel.coord_x
    assert captured["text_coord_y"] == panel.coord_y
    # max_width leaves a `margin`-wide gutter on each side of the panel.
    assert captured["max_width"] == panel.bb_width - panel.config.margin * 2

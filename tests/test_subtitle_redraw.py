from tkinter import Tk

from vspeech.config import Anchor
from vspeech.config import SubtitleTextConfig
from vspeech.lib.subtitle_state import Text
from vspeech.lib.subtitle_state import Texts
from vspeech.worker import subtitle_tk as subtitle_tk_mod
from vspeech.worker.subtitle_tk import draw_text_with_outline
from vspeech.worker.subtitle_tk import redraw_panel


class FakeCanvas:
    """Records the Tk calls redraw_panel makes, without needing a display."""

    def __init__(self, events: list):
        self.events = events

    def delete(self, tag: str):
        self.events.append(("delete", tag))

    def pack(self):
        self.events.append(("pack",))

    def create_text(self, *args, **kwargs):
        # Unused: draw_text_with_outline (the only caller) is monkeypatched out
        # in these tests. Present only so FakeCanvas satisfies redraw_panel's
        # `_SubtitleCanvas` parameter type.
        self.events.append(("create_text", args, kwargs))


def make_panel(
    anchor: Anchor = "s", bb_width: int = 300, bb_height: int = 200
) -> Texts:
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

    monkeypatch.setattr(subtitle_tk_mod, "draw_text_with_outline", fake_draw)
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


def test_draw_text_with_outline_justifies_center_anchor_as_center():
    # This is the only test in the file that calls the real
    # draw_text_with_outline (every other test here monkeypatches it away),
    # so it is what actually proves the justify rule. "center" contains "e"
    # as a substring, so an unguarded `"e" in anchor` test mis-fires and
    # would justify it "right" instead of "center".
    events: list = []
    canvas = FakeCanvas(events)
    root = Tk()
    root.withdraw()
    try:
        draw_text_with_outline(
            canvas=canvas,
            text_coord_x=10,
            text_coord_y=10,
            texts="hello",
            text_tag="text",
            anchor="center",
            config=SubtitleTextConfig(anchor="center"),
        )
    finally:
        root.destroy()

    create_text_calls = [e for e in events if e[0] == "create_text"]
    assert create_text_calls
    for _, _args, kwargs in create_text_calls:
        assert kwargs["justify"] == "center"

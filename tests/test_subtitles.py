"""Subtitles (app/formats/subtitles.py): SRT, WebVTT and ASS."""

from pathlib import Path

import pytest

from app.formats import translate_file
from app.formats.subtitles import read_text
from tests.helpers import run, runner

SRT = """1
00:00:01,000 --> 00:00:03,500
<i>Where are you going
so late at night?</i>

2
00:00:04,000 --> 00:00:06,000
- Home.
- Me too.

3
00:00:07,000 --> 00:00:08,000
♪ ♪
"""


def translate(tmp_path: Path, name: str, content: str | bytes, bilingual: bool = False) -> str:
    src = tmp_path / name
    if isinstance(content, str):
        src.write_text(content, encoding="utf-8", newline="")
    else:
        src.write_bytes(content)
    out = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))[0]
    return out.read_text(encoding="utf-8")


def test_srt_translates_each_cue_and_keeps_timings(tmp_path: Path):
    out = translate(tmp_path, "a.srt", SRT)
    assert out.split("\n") == [
        "1",
        "00:00:01,000 --> 00:00:03,500",
        # The cue's lines are one sentence; italics around the whole cue stay
        "<i>[zh-CN] Where are you going so late at night?</i>",
        "",
        "2",
        "00:00:04,000 --> 00:00:06,000",
        # Two speakers are translated line by line and keep their dashes
        "- [zh-CN] Home.",
        "- [zh-CN] Me too.",
        "",
        "3",
        "00:00:07,000 --> 00:00:08,000",
        "♪ ♪",
        "",
    ]


def test_srt_bilingual_keeps_the_original_above(tmp_path: Path):
    out = translate(tmp_path, "a.srt", SRT, bilingual=True)
    assert "<i>Where are you going\nso late at night?</i>\n<i>[zh-CN] Where are you going so late at night?</i>" in out
    assert "- Home.\n- Me too.\n- [zh-CN] Home.\n- [zh-CN] Me too." in out


def test_srt_keeps_crlf_and_reads_legacy_encodings(tmp_path: Path):
    content = SRT.replace("Me too.", "Café.").replace("♪", "*").replace("\n", "\r\n").encode("cp1252")
    out_path = tmp_path / "a.translated.srt"
    translate(tmp_path, "a.srt", content)
    raw = out_path.read_bytes()
    assert b"\r\n" in raw and b"\n\n" not in raw.replace(b"\r\n", b"")
    assert "- [zh-CN] Café." in raw.decode("utf-8")


@pytest.mark.parametrize(
    "text, codec",
    [
        ("你好，世界。我们明天见面吧。", "gbk"),
        ("你好，世界。我們明天見面吧。", "big5"),
        ("C'est un résumé déjà vu.", "cp1252"),
    ],
)
def test_subtitles_read_legacy_encodings(tmp_path: Path, text: str, codec: str):
    src = tmp_path / "a.srt"
    src.write_bytes(f"1\n00:00:01,000 --> 00:00:02,000\n{text}\n".encode(codec))
    assert read_text(src).split("\n")[2] == text


def test_srt_cues_without_a_blank_line_between_stay_apart(tmp_path: Path):
    srt = "1\n00:00:01,000 --> 00:00:02,000\nHello there.\n2\n00:00:03,000 --> 00:00:04,000\nSee you.\n"
    assert translate(tmp_path, "a.srt", srt).split("\n") == [
        "1",
        "00:00:01,000 --> 00:00:02,000",
        "[zh-CN] Hello there.",
        "2",
        "00:00:03,000 --> 00:00:04,000",
        "[zh-CN] See you.",
        "",
    ]


def test_srt_keeps_override_blocks_at_the_start(tmp_path: Path):
    srt = "1\n00:00:01,000 --> 00:00:02,000\n{\\an8}<i>Hello there,\nmy friend.</i>\n"
    assert translate(tmp_path, "a.srt", srt).split("\n")[2] == "{\\an8}<i>[zh-CN] Hello there, my friend.</i>"


def test_vtt_keeps_header_notes_settings_and_voice(tmp_path: Path):
    vtt = (
        "WEBVTT\n\nNOTE a comment\n\nintro\n00:01.000 --> 00:03.000 align:start\n"
        "<v Anna>Hello &amp; welcome\nto the show.\n\n00:04.000 --> 00:05.000\n<c.yellow>Thanks</c> a lot.\n"
    )
    out = translate(tmp_path, "b.vtt", vtt)
    assert out.startswith("WEBVTT\n\nNOTE a comment\n\nintro\n00:01.000 --> 00:03.000 align:start\n")
    # Entities are decoded for translation and escaped again; the speaker stays
    assert "<v Anna>[zh-CN] Hello &amp; welcome to the show.\n" in out
    assert "\n[zh-CN] Thanks a lot.\n" in out


def test_ass_translates_dialogue_and_keeps_overrides(tmp_path: Path):
    ass = (
        "[Script Info]\nTitle: Demo\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\an8}Good morning,\\Nmy friend.\n"
        "Dialogue: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,{\\p1}m 0 0 l 100 0\n"
        "Comment: 0,0:00:04.00,0:00:05.00,Default,,0,0,0,,Not shown\n"
    )
    out = translate(tmp_path, "c.ass", ass).split("\n")
    assert out[5] == "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\an8}[zh-CN] Good morning, my friend."
    # Vector drawings and comments are left alone
    assert out[6:8] == ass.split("\n")[6:8]
    bilingual = translate(tmp_path, "c.ass", ass, bilingual=True).split("\n")
    assert bilingual[5].endswith("{\\an8}Good morning,\\Nmy friend.\\N[zh-CN] Good morning, my friend.")


def test_subtitles_without_cues_are_rejected(tmp_path: Path):
    src = tmp_path / "empty.srt"
    src.write_text("no cues here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="字幕"):
        run(translate_file(src, tmp_path, runner(), False, "zh-CN"))

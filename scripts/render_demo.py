"""Render an animated terminal demo (GIF and MP4) of one recorded investigation.

Frames are drawn from eval/results.json: the alert, the Planner's hypotheses and checks, the
evidence the Researcher gathered through the MCP tools, each report the Reporter wrote and the
Critic's decision on it. Nothing is typed by hand, so the demo shows exactly what the
evaluation recorded. The free-text summary is left out: the Critic checks the verdict, the fraud
type and the cited evidence, not the prose.

    python scripts/render_demo.py c18 c28      # writes docs/demo.gif and docs/demo.mp4
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
COLS, ROWS, SIZE, PAD = 100, 30, 15, 18
BG, FG = (13, 17, 23), (201, 209, 217)
GREEN, CYAN, YELLOW, GRAY, RED, BLUE = (
    (63, 185, 80),
    (86, 182, 194),
    (210, 168, 75),
    (125, 133, 144),
    (248, 81, 73),
    (121, 192, 255),
)

Color = tuple[int, int, int]
Line = tuple[str, Color]


def wrap(text: str, color: Color, indent: str = "") -> list[Line]:
    out: list[Line] = []
    for para in text.splitlines() or [""]:
        for chunk in textwrap.wrap(para, COLS - len(indent), subsequent_indent="  ") or [""]:
            out.append((indent + chunk, color))
    return out


def script_for(row: dict) -> tuple[str, list[list[Line]]]:
    """The typed command, then blocks of lines revealed one after another."""
    g = row["graph"]
    command = f"$ python scripts/run_eval.py --only {row['id']}"
    evidence = {e["id"]: e for e in g["evidence"]}
    blocks: list[list[Line]] = [wrap(evidence["E1"]["fact"], FG)]
    plan = g["plan"]
    blocks.append(
        [
            ("", FG),
            *wrap(f"planner     hypotheses {', '.join(plan['hypotheses'])}", BLUE),
            *wrap(f"            checks {', '.join(plan['checks'])}", BLUE),
        ]
    )
    for e in g["evidence"][1:]:
        blocks.append(wrap(f"researcher  {e['id']} {e['check']}: {e['fact']}", CYAN))
    critiques = iter(g["critiques"])
    for step in g["trace"]:
        if step["node"] == "reporter":
            blocks.append([("", FG), *wrap(f"reporter    {step['detail']}", YELLOW)])
        elif step["node"] == "critic":
            c = next(critiques)
            if c["approved"]:
                blocks.append(wrap("critic      approved", GREEN))
            else:
                lines = wrap("critic      rejected", RED)
                for p in c["problems"]:
                    lines += wrap(f"- {p}", RED, "            ")
                blocks.append(lines)
    r = g["report"]
    expected = f"{row['expected_verdict']}/{row['expected_type']}"
    got = f"{r['verdict']}/{r['fraud_type']}"
    footer = (
        f"final {got} · expected {expected} · {g['rounds']} round{'s' * (g['rounds'] != 1)} · "
        f"{g['llm_calls']} LLM calls · {g['seconds']:.0f}s on a CPU runner"
    )
    blocks.append([("", FG), (footer, GREEN if got == expected else RED)])
    return command, blocks


class Screen:
    def __init__(self) -> None:
        self.font = ImageFont.truetype(FONT, SIZE)
        box = self.font.getbbox("M")
        self.cw, self.ch = box[2] - box[0], int(SIZE * 1.45)
        self.w, self.h = COLS * self.cw + 2 * PAD, ROWS * self.ch + 2 * PAD
        self.frames: list[tuple[Image.Image, int]] = []

    def draw(self, lines: list[Line], ms: int) -> None:
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        for i, (text, color) in enumerate(lines[-ROWS:]):
            d.text((PAD, PAD + i * self.ch), text, font=self.font, fill=color)
        self.frames.append((img, ms))


def render(rows: list[dict], gif: Path, mp4: Path) -> None:
    screen = Screen()
    for n, row in enumerate(rows):
        command, blocks = script_for(row)
        typed = wrap(command, GREEN)
        screen.draw([], 400)
        for i in range(0, len(command) + 1, 4):  # type the command
            screen.draw(wrap(command[:i] + "▌", GREEN), 35)
        lines = typed[:]
        screen.draw(lines, 500)
        for block in blocks:
            lines += block
            screen.draw(lines, 900)
        screen.draw(lines, 5000 if n == len(rows) - 1 else 3500)

    gif.parent.mkdir(parents=True, exist_ok=True)
    images = [f for f, _ in screen.frames]
    images[0].save(
        gif,
        save_all=True,
        append_images=images[1:],
        loop=0,
        optimize=True,
        duration=[ms for _, ms in screen.frames],
    )
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            concat = Path(tmp) / "frames.txt"
            entries = []
            for i, (img, ms) in enumerate(screen.frames):
                path = Path(tmp) / f"{i:05d}.png"
                img.save(path)
                entries.append(f"file '{path}'\nduration {ms / 1000:.3f}")
            entries.append(f"file '{path}'")
            concat.write_text("\n".join(entries), encoding="utf-8")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                 str(concat), "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset",
                 "veryfast", "-crf", "23", str(mp4)],
                check=True,
            )  # fmt: skip


def main() -> None:
    ids = sys.argv[1:] or ["c18", "c28"]
    results = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in results["rows"]}
    render([by_id[i] for i in ids], ROOT / "docs" / "demo.gif", ROOT / "docs" / "demo.mp4")
    print("wrote docs/demo.gif and docs/demo.mp4")


if __name__ == "__main__":
    main()

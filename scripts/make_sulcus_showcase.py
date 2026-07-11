from __future__ import annotations

import math
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "showcase"
VOICE = OUT / "voice"
FFMPEG = ROOT / ".video_tools" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
GIF_PATH = ROOT / "docs" / "AgentOS_demo.gif"

W, H = 1920, 1080
FPS = 24
DURATION = 72.0
TOTAL_FRAMES = int(DURATION * FPS)

BG = (7, 11, 18)
PANEL = (14, 23, 36)
PANEL_2 = (18, 31, 47)
WHITE = (236, 244, 255)
MUTED = (142, 162, 186)
CYAN = (67, 213, 255)
BLUE = (64, 117, 255)
MINT = (56, 232, 176)
AMBER = (255, 184, 76)
RED = (255, 92, 114)

FONT_REG = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_MONO = Path(r"C:\Windows\Fonts\CascadiaMono.ttf")
if not FONT_MONO.exists():
    FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(str(path), size)


F12 = font(22)
F16 = font(26)
F18 = font(30)
F20 = font(34)
F24 = font(40, bold=True)
F30 = font(50, bold=True)
F36 = font(60, bold=True)
F46 = font(76, bold=True)
F64 = font(108, bold=True)
MONO16 = font(25, mono=True)
MONO18 = font(29, mono=True)
MONO20 = font(33, mono=True)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def ease(v: float) -> float:
    v = clamp(v)
    return v * v * (3 - 2 * v)


def scene_alpha(t: float, start: float, end: float, fade: float = 0.75) -> float:
    return min(ease((t - start) / fade), ease((end - t) / fade))


def rgba(color: tuple[int, int, int], a: int = 255) -> tuple[int, int, int, int]:
    return (*color, a)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def center_text(draw: ImageDraw.ImageDraw, y: int, text: str, fnt: ImageFont.FreeTypeFont, fill=WHITE) -> None:
    x = (W - text_width(draw, text, fnt)) // 2
    draw.text((x, y), text, font=fnt, fill=fill)


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color=CYAN, fnt=F12) -> int:
    tw = text_width(draw, text, fnt)
    width = tw + 34
    draw.rounded_rectangle((x, y, x + width, y + 46), radius=23, fill=rgba(color, 28), outline=rgba(color, 115), width=2)
    draw.text((x + 17, y + 8), text, font=fnt, fill=rgba(color))
    return width


def glass_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], *, accent=CYAN, radius=24) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 8, y1 + 12, x2 + 8, y2 + 12), radius=radius, fill=(0, 0, 0, 85))
    draw.rounded_rectangle(box, radius=radius, fill=rgba(PANEL, 238), outline=rgba(accent, 75), width=2)
    draw.line((x1 + radius, y1 + 2, x2 - radius, y1 + 2), fill=rgba(accent, 150), width=3)


def brand_mark(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float = 1.0, alpha: int = 255) -> None:
    colors = [BLUE, CYAN, MINT, CYAN, BLUE]
    for i, color in enumerate(colors):
        radius = (76 - i * 11) * scale
        pts = []
        for n in range(80):
            a = math.pi * (0.18 + 1.64 * n / 79)
            wobble = math.sin(a * 3 + i * 0.7) * 5 * scale
            x = cx + math.cos(a) * (radius + wobble)
            y = cy + math.sin(a) * radius * 0.72
            pts.append((x, y))
        draw.line(pts, fill=rgba(color, max(60, alpha - i * 22)), width=max(2, int(5 * scale)), joint="curve")


def background(t: float) -> Image.Image:
    img = Image.new("RGBA", (W, H), rgba(BG))
    atmosphere = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atmosphere, "RGBA")
    # Radial-like atmospheric bands.
    for i in range(12, 0, -1):
        r = i * 95
        a = int(3 + (12 - i) * 0.7)
        draw.ellipse((W * 0.70 - r, H * 0.25 - r, W * 0.70 + r, H * 0.25 + r), fill=rgba(BLUE, a))
    # Quiet technical grid.
    offset = int((t * 8) % 80)
    for x in range(-80 + offset, W + 80, 80):
        draw.line((x, 0, x, H), fill=(80, 122, 170, 14), width=1)
    for y in range(-80 + offset, H + 80, 80):
        draw.line((0, y, W, y), fill=(80, 122, 170, 14), width=1)
    # Moving pinpoints keep the background alive without distracting.
    for i in range(22):
        px = int((137 * i + t * (10 + i % 4) * 2) % W)
        py = int((83 * i + 211 + math.sin(t * 0.4 + i) * 35) % H)
        a = int(35 + 45 * (0.5 + 0.5 * math.sin(t * 0.8 + i)))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=rgba(CYAN, a))
    return Image.alpha_composite(img, atmosphere)


def blend(base: Image.Image, layer: Image.Image, opacity: float) -> Image.Image:
    if opacity <= 0:
        return base
    comp = Image.alpha_composite(base, layer)
    return Image.blend(base, comp, clamp(opacity))


def scene_hero(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    p = ease(t / 1.5)
    brand_mark(d, W // 2, 300, 1.65 * p, int(255 * p))
    title_y = int(444 + (1 - p) * 36)
    center_text(d, title_y, "SULCUS OS", F64, WHITE)
    center_text(d, 584, "AN OPERATING SYSTEM FOR AGENTS", F20, CYAN)
    center_text(d, 650, "Run them. Observe them. Recover them.", F18, MUTED)
    x = (W - 544) // 2
    x += pill(d, x, 754, "PYTHON CONTROL PLANE", BLUE) + 18
    pill(d, x, 754, "RUST NATIVE CORE", MINT)
    return layer


def scene_pillars(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.text((150, 106), "AGENTS NEED A RUNTIME", font=F46, fill=WHITE)
    d.text((154, 198), "More than prompts. A place to live, work, and recover.", font=F18, fill=MUTED)
    labels = [
        ("PROCESS LIFECYCLE", "start · stop · supervise", CYAN),
        ("STRUCTURED IPC", "typed agent messages", BLUE),
        ("PERSISTENT MEMORY", "hot · warm · cold", MINT),
        ("FAULT RECOVERY", "detect · restart · verify", AMBER),
        ("ISOLATION", "child process + WASM", RED),
        ("TRACE + REPLAY", "every event inspectable", CYAN),
    ]
    card_w, card_h = 490, 222
    gap_x, gap_y = 42, 38
    start_x, start_y = 160, 316
    for i, (title, sub, accent) in enumerate(labels):
        row, col = divmod(i, 3)
        enter = ease((t - 5.1 - i * 0.16) / 0.75)
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y) + int((1 - enter) * 35)
        glass_card(d, (x, y, x + card_w, y + card_h), accent=accent)
        d.ellipse((x + 30, y + 35, x + 76, y + 81), fill=rgba(accent, 32), outline=rgba(accent, 180), width=3)
        d.ellipse((x + 47, y + 52, x + 59, y + 64), fill=rgba(accent))
        d.text((x + 98, y + 32), title, font=F16, fill=WHITE)
        d.text((x + 32, y + 111), sub, font=F16, fill=MUTED)
        d.line((x + 32, y + 173, x + card_w - 32, y + 173), fill=rgba(accent, 48), width=2)
        status = "RUNTIME READY"
        d.text((x + 32, y + 181), status, font=F12, fill=rgba(accent))
    return layer


def load_demo_frames() -> list[Image.Image]:
    gif = Image.open(GIF_PATH)
    frames: list[Image.Image] = []
    for i in range(150, gif.n_frames):
        gif.seek(i)
        frame = gif.convert("RGB")
        frame = ImageEnhance.Contrast(frame).enhance(1.10)
        frames.append(frame)
    return frames


def rounded_paste(base: Image.Image, src: Image.Image, box: tuple[int, int, int, int], radius: int = 28) -> None:
    x1, y1, x2, y2 = box
    size = (x2 - x1, y2 - y1)
    shot = src.resize(size, Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1, percent=125, threshold=2))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    base.paste(shot, (x1, y1), mask)


DEMO_FRAMES: list[Image.Image] = []


def scene_dashboard(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.text((145, 78), "LIVE CONTROL PLANE", font=F36, fill=WHITE)
    d.text((149, 154), "The dashboard is reading the real Sulcus OS runtime.", font=F16, fill=MUTED)
    box = (145, 238, 1775, 938)
    d.rounded_rectangle((box[0] + 15, box[1] + 20, box[2] + 15, box[3] + 20), radius=32, fill=(0, 0, 0, 130))
    phase = clamp((t - 14.0) / 13.0)
    frame_idx = min(len(DEMO_FRAMES) - 1, int(phase * (len(DEMO_FRAMES) - 1)))
    rounded_paste(layer, DEMO_FRAMES[frame_idx], box, 30)
    d.rounded_rectangle(box, radius=30, outline=rgba(CYAN, 105), width=3)
    px = box[0] + 25
    px += pill(d, px, 965, "AGENT TREE", CYAN) + 12
    px += pill(d, px, 965, "PROCESS REGISTRY", BLUE) + 12
    px += pill(d, px, 965, "IPC MAILBOXES", MINT) + 12
    pill(d, px, 965, "MEMORY PAGES", AMBER)
    return layer


def node(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, accent, subtitle: str = "RUNNING", w: int = 250) -> None:
    glass_card(draw, (x - w // 2, y - 66, x + w // 2, y + 66), accent=accent, radius=22)
    draw.ellipse((x - w // 2 + 20, y - 16, x - w // 2 + 40, y + 4), fill=rgba(accent))
    draw.text((x - w // 2 + 54, y - 37), label, font=F16, fill=WHITE)
    draw.text((x - w // 2 + 54, y + 8), subtitle, font=F12, fill=rgba(accent))


def pulse(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], phase: float, color=CYAN) -> None:
    x = a[0] + (b[0] - a[0]) * phase
    y = a[1] + (b[1] - a[1]) * phase
    draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=rgba(color, 60))
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=rgba(color))


def scene_workflow(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.text((120, 72), "SIX AGENTS. ONE TRACEABLE WORKFLOW.", font=F36, fill=WHITE)
    d.text((124, 148), "A real deterministic run — no API keys or external services.", font=F16, fill=MUTED)
    planner = (265, 520)
    workers = [(700, 330), (700, 520), (700, 710)]
    synth = (1180, 520)
    critic = (1620, 520)
    for w in workers:
        d.line((planner, w), fill=rgba(BLUE, 80), width=4)
        d.line((w, synth), fill=rgba(CYAN, 80), width=4)
    d.line((synth, critic), fill=rgba(MINT, 95), width=4)
    node(d, *planner, "Planner", BLUE, "FAN-OUT", 250)
    node(d, *workers[0], "Benefits", CYAN, "3 FINDINGS", 260)
    node(d, *workers[1], "Risks", AMBER, "3 FINDINGS", 260)
    node(d, *workers[2], "Market", MINT, "3 FINDINGS", 260)
    node(d, *synth, "Synthesizer", CYAN, "FAN-IN", 300)
    node(d, *critic, "Critic", MINT, "8.7 / 10", 260)
    local = t - 27.0
    if local < 5.5:
        ph = (local * 0.55) % 1.0
        for i, w in enumerate(workers):
            pulse(d, planner, w, (ph + i * 0.22) % 1.0, BLUE)
    elif local < 10.0:
        ph = (local * 0.52) % 1.0
        for i, w in enumerate(workers):
            pulse(d, w, synth, (ph + i * 0.22) % 1.0, CYAN)
    else:
        pulse(d, synth, critic, (local * 0.5) % 1.0, MINT)
    logs = [
        ("Planner", "3 assignments sent", BLUE),
        ("Researchers", "9 findings returned", CYAN),
        ("Synthesizer", "report created", MINT),
        ("Critic", "quality score 8.7 / 10", AMBER),
    ]
    y = 880
    x = 150
    for i, (name, msg, color) in enumerate(logs):
        visible = ease((local - i * 2.0) / 0.5)
        if visible <= 0:
            continue
        d.rounded_rectangle((x, y, x + 390, y + 72), radius=18, fill=rgba(PANEL_2, 235), outline=rgba(color, 80), width=2)
        d.text((x + 18, y + 11), name.upper(), font=F12, fill=rgba(color))
        d.text((x + 18, y + 39), msg, font=F12, fill=MUTED)
        x += 410
    return layer


def scene_recovery(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.text((130, 72), "FAILURE IS A RUNTIME EVENT", font=F36, fill=WHITE)
    d.text((134, 149), "Sulcus detects, restarts, and verifies supervised agents.", font=F16, fill=MUTED)
    glass_card(d, (130, 255, 1015, 880), accent=AMBER, radius=28)
    d.text((175, 292), "SUPERVISION TRACE", font=F20, fill=WHITE)
    events = [
        ("00:00.0", "PID 103", "AgentOSDemoCrashProbe", "RUNNING", MINT),
        ("00:02.4", "PID 103", "intentional crash", "CRASHED", RED),
        ("00:02.5", "SUPERVISOR", "restart policy matched", "RESTARTING", AMBER),
        ("00:02.8", "PID 105", "replacement agent", "RUNNING", MINT),
        ("00:03.1", "VERIFY", "replacement replied ok", "RECOVERED", CYAN),
    ]
    local = t - 41.0
    for i, (ts, who, desc, status, color) in enumerate(events):
        y = 390 + i * 92
        active = ease((local - i * 1.35) / 0.45)
        d.line((215, y + 26, 215, y + 92), fill=rgba(color, 70), width=4)
        d.ellipse((203, y + 12, 227, y + 36), fill=rgba(color, int(255 * active) if active else 65))
        d.text((250, y), ts, font=MONO16, fill=MUTED)
        d.text((390, y), who, font=MONO16, fill=rgba(color))
        d.text((555, y), desc, font=F12, fill=WHITE)
        d.text((800, y), status, font=F12, fill=rgba(color))
    glass_card(d, (1075, 255, 1790, 880), accent=CYAN, radius=28)
    d.text((1120, 292), "PROCESS REGISTRY", font=F20, fill=WHITE)
    cols = [(1120, "PID"), (1240, "AGENT"), (1575, "STATUS")]
    for x, label in cols:
        d.text((x, 380), label, font=F12, fill=MUTED)
    rows = [
        ("100", "Supervisor", "running", MINT),
        ("101", "Memory", "running", MINT),
        ("102", "Worker", "isolated", CYAN),
        ("103", "CrashProbe", "crashed", RED),
        ("104", "Coordinator", "running", MINT),
        ("105", "CrashProbe", "restart #1", AMBER),
    ]
    for i, (pid, name, status, color) in enumerate(rows):
        y = 437 + i * 66
        if pid == "105" and local < 6.4:
            color = MUTED
            status = "pending"
        d.rounded_rectangle((1105, y - 7, 1760, y + 48), radius=12, fill=rgba(PANEL_2, 210))
        d.text((1120, y), pid, font=MONO16, fill=WHITE)
        d.text((1240, y), name, font=F12, fill=WHITE)
        d.text((1575, y), status.upper(), font=F12, fill=rgba(color))
    return layer


def scene_features(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    d.text((135, 76), "STATE THAT SURVIVES. EVENTS YOU CAN EXPLAIN.", font=F36, fill=WHITE)
    d.text((139, 152), "The runtime keeps the evidence, not just the final answer.", font=F16, fill=MUTED)
    cards = [(135, 280, 680, 875), (715, 280, 1260, 875), (1295, 280, 1840, 875)]
    accents = [MINT, CYAN, AMBER]
    for box, accent in zip(cards, accents):
        glass_card(d, box, accent=accent, radius=28)
    d.text((180, 330), "PERSISTENT MEMORY", font=F20, fill=MINT)
    d.text((180, 390), "HOT  →  WARM  →  COLD", font=MONO16, fill=MUTED)
    d.rounded_rectangle((180, 488, 635, 675), radius=20, fill=rgba((8, 16, 25), 255), outline=rgba(MINT, 70), width=2)
    d.text((210, 523), "durable-fact", font=MONO16, fill=MINT)
    d.text((210, 581), "worker doubled 21 to 42", font=MONO16, fill=WHITE)
    d.text((180, 745), "paged_memory: 1", font=MONO16, fill=MUTED)

    d.text((760, 330), "STRUCTURED TIMELINE", font=F20, fill=CYAN)
    timeline = [
        ("01", "process_started"),
        ("02", "ipc_message_sent"),
        ("03", "memory_paged"),
        ("04", "process_restarted"),
        ("05", "recovery_verified"),
    ]
    for i, (n, label) in enumerate(timeline):
        y = 432 + i * 72
        d.ellipse((775, y, 807, y + 32), fill=rgba(CYAN, 45), outline=rgba(CYAN), width=2)
        d.text((783, y + 3), n, font=F12, fill=CYAN)
        d.text((832, y), label, font=MONO16, fill=WHITE)
        if i < len(timeline) - 1:
            d.line((791, y + 33, 791, y + 72), fill=rgba(CYAN, 80), width=3)

    d.text((1340, 330), "EXECUTION MODES", font=F20, fill=AMBER)
    modes = [
        ("IN-PROCESS", "full registry attachment", BLUE),
        ("ISOLATED CHILD", "spawned process + IPC bridge", CYAN),
        ("WASM TOOL", "fuel + memory limits", AMBER),
    ]
    for i, (name, sub, color) in enumerate(modes):
        y = 430 + i * 136
        d.rounded_rectangle((1340, y, 1795, y + 106), radius=18, fill=rgba(PANEL_2, 230), outline=rgba(color, 70), width=2)
        d.ellipse((1370, y + 30, 1394, y + 54), fill=rgba(color))
        d.text((1420, y + 17), name, font=F16, fill=WHITE)
        d.text((1420, y + 58), sub, font=F12, fill=MUTED)
    return layer


def scene_final(t: float) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    local = t - 63.0
    p = ease(local / 1.2)
    brand_mark(d, W // 2, 275, 1.35 * p, int(255 * p))
    center_text(d, 415, "SULCUS OS", F64, WHITE)
    center_text(d, 560, "Build agent systems you can see, understand, and trust.", F24, MUTED)
    x = (W - 780) // 2
    d.rounded_rectangle((x, 683, x + 780, 763), radius=40, fill=rgba(PANEL_2, 245), outline=rgba(CYAN, 100), width=2)
    d.text((x + 43, 701), "github.com/ElarizT/SulcusOS", font=MONO20, fill=CYAN)
    center_text(d, 840, "PYTHON + RUST   •   NATIVE IPC   •   WASM ISOLATION", F16, WHITE)
    center_text(d, 930, "EXPERIMENTAL AGENT RUNTIME", F12, MUTED)
    return layer


SCENES = [
    (0.0, 6.0, scene_hero),
    (5.0, 15.0, scene_pillars),
    (14.0, 28.0, scene_dashboard),
    (27.0, 42.0, scene_workflow),
    (41.0, 54.0, scene_recovery),
    (53.0, 64.0, scene_features),
    (63.0, 72.0, scene_final),
]


def render_frame(t: float) -> Image.Image:
    frame = background(t)
    for start, end, fn in SCENES:
        alpha = scene_alpha(t, start, end)
        if alpha > 0:
            frame = blend(frame, fn(t), alpha)
    # Cinematic edge shading.
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade, "RGBA")
    sd.rectangle((0, 0, W, 42), fill=(0, 0, 0, 80))
    sd.rectangle((0, H - 34, W, H), fill=(0, 0, 0, 90))
    return Image.alpha_composite(frame, shade).convert("RGB")


def make_music(path: Path) -> None:
    sample_rate = 48_000
    n = int(DURATION * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    audio = np.zeros(n, dtype=np.float64)
    chords = [
        (65.41, 98.00, 155.56),
        (51.91, 103.83, 155.56),
        (77.78, 116.54, 174.61),
        (58.27, 87.31, 130.81),
    ]
    block = 8.0
    for i, start in enumerate(np.arange(0, DURATION, block)):
        end = min(DURATION, start + block)
        mask = (t >= start) & (t < end)
        tt = t[mask] - start
        chord = chords[i % len(chords)]
        env = np.minimum(1.0, tt / 1.2) * np.minimum(1.0, (end - start - tt) / 1.3)
        pad = sum(np.sin(2 * np.pi * f * tt + j * 0.7) for j, f in enumerate(chord)) / len(chord)
        shimmer = np.sin(2 * np.pi * chord[1] * 2 * tt + 0.7 * np.sin(tt * 0.8))
        audio[mask] += env * (0.13 * pad + 0.018 * shimmer)
    # Gentle pulse and transition pings.
    for beat in np.arange(1.0, DURATION, 2.0):
        mask = (t >= beat) & (t < beat + 0.35)
        tt = t[mask] - beat
        audio[mask] += 0.045 * np.sin(2 * np.pi * 58 * tt) * np.exp(-tt * 10)
    for hit in [5, 14, 27, 41, 53, 63]:
        mask = (t >= hit) & (t < hit + 1.4)
        tt = t[mask] - hit
        audio[mask] += 0.035 * np.sin(2 * np.pi * 740 * tt) * np.exp(-tt * 3.2)
    audio *= np.minimum(1.0, t / 2.0) * np.minimum(1.0, (DURATION - t) / 3.0)
    audio = np.tanh(audio * 1.4)
    stereo = np.stack([audio * 0.95, audio], axis=1)
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def render_video(silent_path: Path) -> None:
    cmd = [
        str(FFMPEG), "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(TOTAL_FRAMES):
            frame = render_frame(i / FPS)
            proc.stdin.write(frame.tobytes())
            if i % (FPS * 5) == 0:
                print(f"Rendered {i / FPS:5.1f}s / {DURATION:.1f}s", flush=True)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
        if code != 0:
            raise RuntimeError(stderr[-5000:])
    finally:
        if proc.poll() is None:
            proc.kill()


def mux_audio(silent_path: Path, final_path: Path) -> None:
    starts_ms = [500, 6000, 16200, 29600, 43300, 54800, 65400]
    voice_files = [
        VOICE / "01_intro.wav",
        VOICE / "02_runtime.wav",
        VOICE / "03_dashboard.wav",
        VOICE / "04_research.wav",
        VOICE / "05_recovery.wav",
        VOICE / "06_memory.wav",
        VOICE / "07_close.wav",
    ]
    cmd = [str(FFMPEG), "-y", "-i", str(silent_path), "-i", str(OUT / "music.wav")]
    for path in voice_files:
        cmd.extend(["-i", str(path)])
    filters = ["[1:a]volume=0.52[music]"]
    labels = ["[music]"]
    for i, delay in enumerate(starts_ms, start=2):
        label = f"v{i}"
        filters.append(f"[{i}:a]volume=1.35,adelay={delay}|{delay}[{label}]")
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0,alimiter=limit=0.95[aout]")
    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "192k", "-t", str(DURATION), "-movflags", "+faststart", str(final_path),
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-5000:])


def main() -> None:
    global DEMO_FRAMES
    OUT.mkdir(parents=True, exist_ok=True)
    if not FFMPEG.exists():
        raise FileNotFoundError(f"ffmpeg was not found at {FFMPEG}")
    missing_voice = [p.name for p in VOICE.glob("*.wav")]
    if len(missing_voice) < 7:
        raise RuntimeError("Narration files are missing. Run scripts/render_showcase_voice.ps1 first.")
    print("Loading authentic dashboard capture...", flush=True)
    DEMO_FRAMES = load_demo_frames()
    make_music(OUT / "music.wav")
    render_frame(1.8).save(OUT / "sulcus_os_showcase_poster.png")
    render_frame(38.0).save(OUT / "sulcus_os_showcase_workflow.png")
    silent = OUT / "sulcus_os_showcase_silent.mp4"
    final = OUT / "sulcus_os_showcase.mp4"
    render_video(silent)
    mux_audio(silent, final)
    print(f"Final video: {final}", flush=True)


if __name__ == "__main__":
    main()

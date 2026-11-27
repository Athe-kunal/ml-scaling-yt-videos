"""Render only the lecture sections whose code changed — in parallel — then present.

    python render.py                     # render what changed (auto: sized to the main screen), all lessons
    python render.py L2 --quality=-l     # one lesson, fast 480p
    python render.py L2_Main_03 --present   # rehearse a single section
    python render.py --present | --html  # render, then open the player / browser version

Each lesson is split into sections (Lesson.SECTIONS); every section is its own small
scene. A section's fingerprint = its own method's source + the rest of its lesson file
(minus the other sections) + every helper module (style.py, toy.py, …) + quality. Edit one section → only that
section re-renders; edit shared code → the whole lesson (or everything) re-renders.
"""
import argparse
import hashlib
import re
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

# (module, Lesson class) in lecture order — fill this in as lessons are written
LESSONS = [("l1_intro", "L1_Intro")]
LESSONS = [x for x in LESSONS if (HERE / f"{x[0]}.py").exists()]
STAMPS = HERE / "slides" / ".stamps"
BIN = Path(sys.executable).parent / "manim-slides"


def sections():
    """[(module, scene_name, fingerprint)] in lecture order."""
    # every helper module (style.py, toy.py, widgets…) is shared: editing one re-renders everything
    shared = "".join(p.read_text() for p in sorted(HERE.glob("*.py"))
                     if not re.match(r"l\d+_", p.name) and p.name not in ("render.py", "sidebar.py"))
    from style import section_scene_name
    out = []
    for mod_name, cls_name in LESSONS:
        mod = importlib.import_module(mod_name)
        lesson = getattr(mod, cls_name)
        file_src = (HERE / f"{mod_name}.py").read_text()
        srcs = {s: inspect.getsource(getattr(lesson, s)) for s in lesson.SECTIONS}
        for i, s in enumerate(lesson.SECTIONS):
            rest = file_src
            for other, src in srcs.items():
                if other != s and src in rest:
                    rest = rest.replace(src, "")
            fp = hashlib.sha1((srcs[s] + rest + shared).encode()).hexdigest()
            out.append((mod_name, section_scene_name(lesson, i, s), fp))
    return out


def main_screen_px():
    """Physical pixels (width, height) of the main screen, or None."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True,
                                 text=True, timeout=10).stdout
            last, first = None, None
            for line in out.splitlines():
                m = re.search(r"Resolution:\s*(\d+)\s*x\s*(\d+)", line)
                if m:
                    last = (int(m[1]), int(m[2]))
                    first = first or last
                if "Main Display: Yes" in line and last:
                    return last
            return first
        if shutil.which("xrandr"):
            out = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"primary (\d+)x(\d+)\+", out) or re.search(r" connected (\d+)x(\d+)\+", out)
            if m:
                return int(m[1]), int(m[2])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def screen_quality():
    """Render size = the largest 16:9 frame that fits the main screen's physical pixels.

    Video is raster: smaller than the screen → upscaled and soft; bigger → the player has to
    shrink it (and some players mis-place oversized video). Matching the screen avoids both:
    Retina 3456×2234 → 3456×1944, 4K → 3840×2160, 1080p → 1920×1080. Capped at 4K, floor 720p.
    """
    px = main_screen_px()
    if not px:
        return "--hd"
    k = min(px[0] // 16, px[1] // 9, 240)
    k = max(80, k - k % 2)                    # even height for H.264
    return f"--resolution={16 * k}x{9 * k}"


def render_one(mod_name, scene, quality):
    shutil.rmtree(HERE / "videos" / scene, ignore_errors=True)  # stale *_temp.mp4 breaks manim-slides
    t0 = time.time()
    # own temp dir per process: ManimGL writes text SVGs to <tmp>/<hash>.svg and deletes them, so
    # parallel renders of the same label (e.g. the roadmap) would otherwise delete each other's files
    tmp = HERE / "media" / "tmp" / scene
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": str(tmp)}
    p = subprocess.run([str(BIN), "render", "--GL", f"{mod_name}.py", scene, quality],
                       capture_output=True, text=True, env=env)
    log = (p.stdout + p.stderr).replace("\r", "\n")
    warn = [l for l in log.splitlines() if l.startswith("SAFE-AREA")]
    ok = p.returncode == 0 and (HERE / "slides" / f"{scene}.json").exists()
    tail = "\n".join(l for l in log.splitlines()[-25:] if "it/s" not in l and "slides/s" not in l) if not ok else ""
    return scene, ok, time.time() - t0, warn, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="filters on scene names, e.g. L2, L3_LogMel, probe")
    ap.add_argument("--quality", default="auto",
                    help="auto (match the main screen, e.g. 3456x1944 on Retina) or a manimgl flag: -l, -m, --hd, --uhd")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--present", action="store_true", help="fullscreen browser presentation (fits any screen)")
    ap.add_argument("--qt", action="store_true", help="with --present: the manim-slides Qt player instead")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="with --html: write the page but don't open it")
    ap.add_argument("--force", action="store_true", help="re-render even if unchanged")
    a = ap.parse_args()
    if a.quality == "auto":
        a.quality = screen_quality()
    sys.argv = sys.argv[:1]  # manimlib parses sys.argv when the lessons import it

    allsec = sections()
    chosen = [x for x in allsec if not a.only or any(f in x[1] for f in a.only)]
    if not chosen:
        sys.exit(f"no section matches {a.only}")
    STAMPS.mkdir(parents=True, exist_ok=True)

    def stale(mod_name, scene, fp):
        stamp = STAMPS / scene
        want = f"{fp} {a.quality}"
        return a.force or not (HERE / "slides" / f"{scene}.json").exists() or \
            not stamp.exists() or stamp.read_text() != want

    todo = [x for x in chosen if stale(*x)]
    print(f"{len(chosen)} sections · {len(todo)} to render · {a.jobs} in parallel · quality {a.quality}", flush=True)
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(render_one, m, s, a.quality): (m, s, fp) for m, s, fp in todo}
        for f in as_completed(futs):
            m, s, fp = futs[f]
            scene, ok, dt, warn, tail = f.result()
            print(f"  {'ok ' if ok else 'ERR'} {scene:<34} {dt:5.1f}s", flush=True)
            for w in warn:
                print("     " + w, flush=True)
            if ok:
                (STAMPS / scene).write_text(f"{fp} {a.quality}")
            else:
                failed.append(scene)
                print(tail, flush=True)
    if todo:
        print(f"rendered in {time.time() - t0:.0f}s", flush=True)
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")

    scenes = [s for _, s, _ in chosen]
    if a.present and a.qt:
        subprocess.run([str(BIN), "present", *scenes, "--full-screen", "--aspect-ratio", "keep"])
    if a.html or (a.present and not a.qt):
        out = write_html(scenes)
        if a.present:
            open_fullscreen(out)
        elif not a.no_open:
            import webbrowser
            webbrowser.open(out.as_uri())


def write_html(scenes):
    """All sections as one reveal.js page (video scaled to fit any window) + contents sidebar."""
    from sidebar import inject
    from style import section_scene_name
    out = HERE / "slides" / "lectures.html"
    subprocess.run([str(BIN), "convert", *scenes, str(out)], check=True)
    entries = []  # sidebar: one heading per lesson, one link per subtopic (its roadmap beat)
    for mod_name, cls_name in LESSONS:
        lesson = getattr(importlib.import_module(mod_name), cls_name)
        sc = [(section_scene_name(lesson, i, s), lesson.TOPICS.get(s)) for i, s in enumerate(lesson.SECTIONS)]
        sc = [x for x in sc if x[0] in scenes]
        if sc:
            entries.append((f"{lesson.NUM} · {lesson.TITLE}", sc))
    inject(out, entries, HERE / "slides")
    print(f"wrote {out}", flush=True)
    return out


def open_fullscreen(page):
    """Present in a borderless fullscreen browser window: the browser fits and centres the
    video on whatever screen it is on (any size, any pixel density). Exit: ⌘Q / Alt+F4."""
    url = page.as_uri()
    chrome = {"darwin": ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"],
              "linux": [shutil.which("google-chrome") or "", shutil.which("chromium") or ""]}.get(sys.platform, [])
    exe = next((c for c in chrome if c and Path(c).exists()), None)
    if exe:
        subprocess.Popen([exe, f"--app={url}", "--start-fullscreen", "--autoplay-policy=no-user-gesture-required",
                          f"--user-data-dir={HERE / 'slides' / '.browser'}", "--no-first-run",
                          "--no-default-browser-check"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("presenting fullscreen in the browser: ← → step · T contents · ⌘Q quit", flush=True)
    else:
        import webbrowser
        webbrowser.open(url)
        print("opened in the browser: press F for fullscreen · ← → step · T contents", flush=True)


if __name__ == "__main__":
    main()

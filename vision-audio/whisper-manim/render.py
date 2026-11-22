"""Render only the lecture sections whose code changed — in parallel — then present.

    python render.py                     # render what changed (1080p), all lessons
    python render.py L2 --quality=-l     # one lesson, fast 480p
    python render.py L2_Fourier_02 --present   # rehearse a single section
    python render.py --present | --html  # render, then open the player / browser version

Each lesson is split into sections (Lesson.SECTIONS); every section is its own small
scene. A section's fingerprint = its own method's source + the rest of its lesson file
(minus the other sections) + style.py + dsp.py + quality. Edit one section → only that
section re-renders; edit shared code → the whole lesson (or everything) re-renders.
"""
import argparse
import hashlib
import importlib
import inspect
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

LESSONS = [("l1_waveform", "L1_Waveform"), ("l2_fourier", "L2_Fourier"),
           ("l3_logmel", "L3_LogMel"), ("l5_encoder", "L5_Encoder")]
STAMPS = HERE / "slides" / ".stamps"
BIN = Path(sys.executable).parent / "manim-slides"


def sections():
    """[(module, scene_name, fingerprint)] in lecture order."""
    shared = (HERE / "style.py").read_text() + (HERE / "dsp.py").read_text()
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


def render_one(mod_name, scene, quality):
    shutil.rmtree(HERE / "videos" / scene, ignore_errors=True)  # stale *_temp.mp4 breaks manim-slides
    t0 = time.time()
    p = subprocess.run([str(BIN), "render", "--GL", f"{mod_name}.py", scene, quality],
                       capture_output=True, text=True)
    log = (p.stdout + p.stderr).replace("\r", "\n")
    warn = [l for l in log.splitlines() if l.startswith("SAFE-AREA")]
    ok = p.returncode == 0 and (HERE / "slides" / f"{scene}.json").exists()
    tail = "\n".join(l for l in log.splitlines()[-25:] if "it/s" not in l and "slides/s" not in l) if not ok else ""
    return scene, ok, time.time() - t0, warn, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="filters on scene names, e.g. L2, L3_LogMel, probe")
    ap.add_argument("--quality", default="--hd", help="manimgl flag: -l (480p), -m (720p), --hd (1080p)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--present", action="store_true")
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render even if unchanged")
    a = ap.parse_args()
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
    if a.present:
        subprocess.run([str(BIN), "present", *scenes, "--full-screen", "--aspect-ratio", "keep"])
    if a.html:
        subprocess.run([str(BIN), "convert", *scenes, "slides/lectures.html", "--open"])


if __name__ == "__main__":
    main()

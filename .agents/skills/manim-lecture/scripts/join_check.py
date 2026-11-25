"""Check that consecutive section scenes join seamlessly (last frame of A ≈ first frame of B).

    python join_check.py <app-dir>            # all consecutive pairs without a roadmap in between
Prints SSIM per join; anything < 0.98 is a visible jump (state rebuilt differently, fade from blank…).
Pairs where B starts with a roadmap slide are skipped (those are meant to cut).
"""
import json, subprocess, sys, tempfile
from pathlib import Path

app = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(app))
sys.argv = sys.argv[:1]
import render  # noqa: E402  (chdirs into the app)
import importlib  # noqa: E402

tmp = Path(tempfile.mkdtemp())
for mod_name, cls_name in render.LESSONS:
    lesson = getattr(importlib.import_module(mod_name), cls_name)
    from style import section_scene_name
    names = [(section_scene_name(lesson, i, s), s) for i, s in enumerate(lesson.SECTIONS)]
    for (a, _), (b, sec_b) in zip(names, names[1:]):
        if sec_b in lesson.TOPICS:
            continue
        ja, jb = app / "slides" / f"{a}.json", app / "slides" / f"{b}.json"
        if not (ja.exists() and jb.exists()):
            print(f"  ?    {a} → {b}: not rendered"); continue
        fa = json.load(open(ja))["slides"][-1]["file"]; fb = json.load(open(jb))["slides"][0]["file"]
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.05", "-i", fa, "-frames:v", "1", tmp / "a.png"])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", fb, "-frames:v", "1", tmp / "b.png"])
        r = subprocess.run(["ffmpeg", "-i", tmp / "a.png", "-i", tmp / "b.png", "-lavfi", "ssim", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
        ssim = float([l for l in r.splitlines() if "All:" in l][-1].split("All:")[1].split()[0])
        print(f"  {'ok ' if ssim >= 0.98 else 'JUMP'} {ssim:.3f}  {a} → {b}")

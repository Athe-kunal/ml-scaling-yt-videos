"""Clickable table of contents for the reveal.js version of the lecture.

manim-slides writes every beat as one flat <section>, so "slide h" = the h-th beat of the
lecture. We count each section scene's beats from slides/<scene>.json and inject a sidebar
whose links call Reveal.slide(h). T toggles it; the current section is highlighted.
"""
import html
import json
from pathlib import Path

CSS = """
#toc-btn{position:fixed;top:14px;left:14px;z-index:30;width:40px;height:40px;border-radius:8px;
  border:1px solid #334053;background:#161D29cc;color:#C7D1DD;font:20px/38px 'Helvetica Neue',sans-serif;
  text-align:center;cursor:pointer}
#toc{position:fixed;top:0;left:0;bottom:0;width:320px;z-index:29;overflow-y:auto;padding:66px 14px 24px;
  background:#0A0E14f2;border-right:1px solid #334053;font-family:'Helvetica Neue',sans-serif;
  transform:translateX(-100%);transition:transform .2s ease}
#toc.open{transform:none}
body.toc-open .reveal{left:320px !important;width:calc(100% - 320px) !important}
html,body{background:#0A0E14}
#toc{scrollbar-width:thin;scrollbar-color:#334053 transparent}
#toc::-webkit-scrollbar{width:8px}#toc::-webkit-scrollbar-thumb{background:#334053;border-radius:4px}
#toc .lesson{display:block;margin:14px 0 4px;padding:6px 10px;border-radius:6px;color:#F5F8FB;
  font-size:17px;font-weight:600;cursor:pointer}
#toc .sec{display:block;padding:5px 10px 5px 26px;border-radius:6px;color:#94A1B2;font-size:15px;cursor:pointer}
#toc .lesson:hover,#toc .sec:hover{background:#161D29}
#toc .sec.active{color:#6EF3DD;background:#6EF3DD18}
#toc .lesson.active{color:#6EF3DD}
#toc .num{color:#94A1B2;margin-right:6px;font-variant-numeric:tabular-nums}
#toc .hint{color:#334053;font-size:12px;padding:0 10px 8px}
"""

JS = """
(function(){
  var toc=document.getElementById('toc'), btn=document.getElementById('toc-btn');
  function toggle(){            // the slide shrinks beside the open sidebar, so the frame stays fully visible
    toc.classList.toggle('open'); document.body.classList.toggle('toc-open');
    setTimeout(function(){ Reveal.layout(); }, 220);
  }
  btn.addEventListener('click', toggle);
  document.addEventListener('keydown', function(e){
    if ((e.key==='t'||e.key==='T') && !e.metaKey && !e.ctrlKey) { toggle(); }
  });
  var links=[].slice.call(toc.querySelectorAll('[data-h]'));
  links.forEach(function(a){ a.addEventListener('click', function(){ Reveal.slide(+a.dataset.h); }); });
  function mark(h){
    var sec=null, les=null;
    links.forEach(function(a){ a.classList.remove('active');
      if (+a.dataset.h<=h) { if (a.classList.contains('lesson')) { les=a; sec=null; } else sec=a; } });
    if (les) les.classList.add('active');
    if (sec) { sec.classList.add('active'); sec.scrollIntoView({block:'nearest'}); }
  }
  Reveal.on('slidechanged', function(e){ mark(e.indexh); });
  Reveal.on('ready', function(e){ mark(e.indexh); });
  if (Reveal.isReady && Reveal.isReady()) { mark(Reveal.getIndices().h); }
})();
"""


def build(entries, slides_dir: Path):
    """entries: [(lesson_label, [(scene, section_label_or_None), ...]), ...] in lecture order.
    Returns the sidebar HTML (links carry the index of the first beat of each scene)."""
    h, rows = 0, []
    for lesson_label, scenes in entries:
        first = True
        for scene, label in scenes:
            n = len(json.loads((slides_dir / f"{scene}.json").read_text())["slides"])
            if first:
                rows.append(f'<a class="lesson" data-h="{h}">{html.escape(lesson_label)}</a>')
                first = False
            if label:
                rows.append(f'<a class="sec" data-h="{h}">{html.escape(label)}</a>')
            h += n
    return ('<div id="toc-btn" title="Contents (T)">☰</div>'
            '<nav id="toc"><div class="hint">click to jump · T to show/hide</div>' + "".join(rows) + "</nav>")


def inject(html_path: Path, entries, slides_dir: Path):
    page = html_path.read_text()
    page = page.replace("</head>", f"<style>{CSS}</style></head>", 1)
    page = page.replace("</body>", build(entries, slides_dir) + f"<script>{JS}</script></body>", 1)
    html_path.write_text(page)

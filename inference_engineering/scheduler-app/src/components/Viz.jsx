import { T, mono } from "../theme";
import { short } from "../sim/trace";

// Pure views over one event snapshot: ev.snap holds the world, ev.* says
// what the current line touched (pages, hashes, request).

const HIT = "#A5D6A7";
const HASH = "#C49CFF";

const colorMap = (snap) => Object.fromEntries(snap.reqs.map((r) => [r.id, r.color]));

export function Pill({ id, color, dim, small }) {
  return (
    <span style={{
      fontFamily: mono, fontSize: small ? 10 : 11.5, fontWeight: 700, padding: small ? "1px 6px" : "2px 8px", borderRadius: 999,
      color: dim ? T.dim : T.bg, background: dim ? "transparent" : color, border: `1px solid ${color}`, whiteSpace: "nowrap",
    }}>{id}</span>
  );
}

export function HashChip({ h, lit, color = HASH, strike }) {
  if (h == null) return <span style={{ fontFamily: mono, fontSize: 10, color: T.rule }}>—</span>;
  return (
    <span style={{
      fontFamily: mono, fontSize: 10, padding: "1px 5px", borderRadius: 4, whiteSpace: "nowrap",
      color: lit ? T.bg : color, background: lit ? color : `${color}14`, border: `1px solid ${color}${lit ? "" : "55"}`,
      textDecoration: strike ? "line-through" : "none", transition: "all .25s",
    }}>#{short(h)}</span>
  );
}

// one token slot
export function Tok({ t, color, state, gen, size = 22, pop }) {
  const s = {
    width: size, height: size, borderRadius: 4, display: "inline-flex", alignItems: "center", justifyContent: "center",
    fontFamily: mono, fontSize: size * 0.52, fontWeight: 600, fontStyle: gen ? "italic" : "normal", flexShrink: 0,
    transition: "background .3s, border-color .3s, color .3s, opacity .3s",
  };
  const st = {
    kv: { background: `${color}38`, border: `1px solid ${color}`, color: T.ink },
    hit: { background: `${HIT}30`, border: `1px solid ${HIT}`, color: HIT },
    sched: { background: `${color}12`, border: `1.5px dashed ${color}`, color },
    pending: { background: "transparent", border: `1px solid ${T.rule}`, color: T.dim },
    empty: { background: "transparent", border: `1px dotted ${T.rule}`, color: "transparent" },
    stale: { background: "transparent", border: `1px solid ${T.rule}`, color: T.dim, opacity: 0.4, textDecoration: "line-through" },
  }[state];
  return <span className={`${state === "sched" ? "tok-pulse" : ""} ${pop ? "tok-pop" : ""}`} style={{ ...s, ...st, "--c": color }}>{t ?? ""}</span>;
}

// ---------------------------------------------------------------- requests

export function RequestRows({ ev, bs, visible }) {
  const snap = ev.snap;
  const reqs = snap.reqs.filter(visible || (() => true));
  if (!reqs.length) return <div style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>no requests yet</div>;
  return (
    <div style={{ display: "grid", gap: 12 }}>
      {reqs.map((r) => {
        const active = ev.req === r.id;
        const done = snap.done.includes(r.id);
        const sb = snap.batch && snap.batch.find((s) => s.rid === r.id);
        const nBlocks = Math.max(Math.ceil(r.token_ids.length / bs), r.block_ids.length);
        const state = (p) => {
          if (p < r.kvWritten) return p < r.hit ? "hit" : "kv";
          if (sb && p >= sb.num_computed && p < sb.num_computed + sb.num_new) return "sched";
          return "pending";
        };
        return (
          <div key={r.id} style={{
            display: "flex", gap: 12, alignItems: "flex-start", padding: "8px 10px", borderRadius: 8,
            background: active ? `${ev.color}10` : "transparent", border: `1px solid ${active ? `${ev.color}66` : "transparent"}`,
            opacity: done ? 0.45 : 1, transition: "all .3s",
          }}>
            <div style={{ width: 54, flexShrink: 0, paddingTop: 3, display: "grid", gap: 4, justifyItems: "start" }}>
              <Pill id={r.id} color={r.color} dim={done} />
              <span style={{ fontFamily: mono, fontSize: 9.5, color: T.dim }}>nc={r.num_computed}</span>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {Array.from({ length: nBlocks }, (_, j) => {
                const page = r.block_ids[j];
                const h = r.block_hashes[j];
                const hlit = (ev.hashBlock && ev.hashBlock.rid === r.id && ev.hashBlock.i === j);
                const plit = page !== undefined && ev.pages && ev.pages[page] && active;
                return (
                  <div key={j} style={{ display: "grid", gap: 4, justifyItems: "center" }}>
                    <div style={{
                      display: "flex", gap: 2, padding: 3, borderRadius: 6,
                      border: `1px solid ${plit ? ev.color : T.rule}`, background: T.well, position: "relative",
                    }}>
                      {Array.from({ length: bs }, (_, o) => {
                        const p = j * bs + o;
                        if (p >= r.token_ids.length) return <Tok key={o} state="empty" />;
                        return (
                          <span key={o} style={{ position: "relative", display: "inline-flex" }}>
                            {p === r.num_computed && <Cursor />}
                            <Tok t={r.token_ids[p]} color={r.color} state={state(p)} gen={p >= r.prompt_len} pop={ev.sampled === p && active} />
                          </span>
                        );
                      })}
                      {r.num_computed === (j + 1) * bs && r.num_computed <= r.token_ids.length && <Cursor right />}
                    </div>
                    <div style={{ display: "flex", gap: 4, alignItems: "center", minHeight: 18 }}>
                      <span style={{
                        fontFamily: mono, fontSize: 10, padding: "1px 5px", borderRadius: 4, transition: "all .25s",
                        color: page === undefined ? T.rule : plit ? T.bg : T.soft,
                        background: plit ? ev.color : page === undefined ? "transparent" : T.panel,
                        border: `1px solid ${page === undefined ? T.rule : plit ? ev.color : T.dim}`,
                      }}>{page === undefined ? "p ?" : `p${page}`}</span>
                      {h && <HashChip h={h} lit={hlit} color={ev.lookup && hlit ? (ev.lookup.hit ? HIT : "#FF8A8A") : HASH} />}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// num_computed_tokens boundary
function Cursor({ right }) {
  return <span style={{ position: "absolute", [right ? "right" : "left"]: right ? 0 : -3, top: -3, bottom: -3, width: 2, borderRadius: 1, background: T.accent, boxShadow: `0 0 6px ${T.accent}` }} />;
}

// ---------------------------------------------------------------- pages

export function PagePool({ ev }) {
  const snap = ev.snap;
  const cm = colorMap(snap);
  const owners = (b) => snap.reqs.filter((r) => r.block_ids.includes(b));
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
      {snap.pages.map((slots, b) => {
        const own = owners(b);
        const lru = snap.free.indexOf(b);
        const act = ev.pages && ev.pages[b];
        const isNull = b === 0;
        const border = act ? ev.color : isNull ? T.rule : own.length ? own[0].color : T.dim;
        return (
          <div key={b} style={{
            position: "relative", padding: "6px 6px 5px", borderRadius: 8, minWidth: 74,
            border: `${act ? 2 : 1}px ${!isNull && !own.length ? "dashed" : "solid"} ${border}`,
            background: isNull ? `repeating-linear-gradient(45deg, ${T.well}, ${T.well} 5px, ${T.panel} 5px, ${T.panel} 10px)` : T.well,
            boxShadow: act ? `0 0 14px ${ev.color}55` : "none", transition: "all .3s", margin: act ? 0 : 1,
          }}>
            {act && <span style={{ position: "absolute", top: -9, right: 6, fontFamily: mono, fontSize: 9.5, fontWeight: 700, padding: "0 5px", borderRadius: 4, background: ev.color, color: T.bg }}>{act}</span>}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6, marginBottom: 5 }}>
              <span style={{ fontFamily: mono, fontSize: 11, fontWeight: 700, color: isNull ? T.dim : T.ink }}>p{b}</span>
              {isNull ? <span style={{ fontFamily: mono, fontSize: 9, color: T.dim }}>null</span>
                : snap.ref[b] > 0 ? <span style={{ fontFamily: mono, fontSize: 9.5, color: border }}>ref {snap.ref[b]}</span>
                  : <span style={{ fontFamily: mono, fontSize: 9.5, color: T.dim }}>LRU {lru + 1}</span>}
            </div>
            <div style={{ display: "flex", gap: 2 }}>
              {slots.map((s, o) => (
                <Tok key={o} size={18} t={s ? s.tok : null} color={s ? cm[s.rid] : T.rule}
                  state={!s ? "empty" : s.stale ? "stale" : "kv"} gen={s && s.gen} />
              ))}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 5, gap: 4, minHeight: 16 }}>
              <HashChip h={snap.bh[b]} lit={act === "hash"} />
              <span style={{ display: "flex", gap: 2 }}>
                {own.map((r) => <span key={r.id} title={r.id} style={{ width: 8, height: 8, borderRadius: 4, background: r.color }} />)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// LRU free list: chips slide as the OrderedDict changes
export function FreeList({ ev }) {
  const snap = ev.snap;
  const W = 46;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, width: 54, flexShrink: 0, lineHeight: 1.3 }}>front<br />popitem</span>
      <div style={{ position: "relative", height: 34, flex: 1, minWidth: 0, borderRadius: 6, background: T.well, border: `1px solid ${T.rule}`, overflow: "hidden" }}>
        {snap.free.map((b, k) => {
          const act = ev.pages && ev.pages[b];
          const cached = snap.bh[b] != null;
          return (
            <span key={b} style={{
              position: "absolute", top: 4, left: 4 + k * W, width: W - 6, height: 24, borderRadius: 5,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 3,
              fontFamily: mono, fontSize: 11, fontWeight: 600, transition: "left .45s ease, background .3s",
              color: act ? T.bg : T.soft, background: act ? ev.color : T.panel, border: `1px solid ${cached ? HASH : T.rule}`,
            }}>
              p{b}{cached && <span style={{ width: 5, height: 5, borderRadius: 3, background: act ? T.bg : HASH }} />}
            </span>
          );
        })}
        {!snap.free.length && <span style={{ position: "absolute", left: 10, top: 9, fontFamily: mono, fontSize: 11, color: "#FF8A8A" }}>empty</span>}
      </div>
      <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, width: 44, flexShrink: 0, lineHeight: 1.3 }}>tail<br />← free</span>
    </div>
  );
}

// ---------------------------------------------------------------- hashes

export function HashTable({ ev, hashInfo }) {
  const snap = ev.snap;
  const rows = snap.h2b.map(([h, b]) => ({ h, b }));
  const extra = [];
  if (ev.lookup && !ev.lookup.hit) extra.push({ h: ev.lookup.h, miss: true });
  for (const h of ev.evicted || []) extra.push({ h, evicted: true });
  if (!rows.length && !extra.length) return <div style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>{"{}"}</div>;
  return (
    <div style={{ display: "grid", gap: 4 }}>
      {[...rows, ...extra].map((row) => {
        const info = hashInfo[row.h] || { parent: null, toks: [] };
        const lit = (ev.hashes && ev.hashes.includes(row.h)) || (ev.lookup && ev.lookup.h === row.h);
        const c = row.miss || row.evicted ? "#FF8A8A" : ev.lookup && ev.lookup.h === row.h ? HIT : HASH;
        return (
          <div key={row.h + (row.miss ? "m" : row.evicted ? "e" : "")} style={{
            display: "flex", alignItems: "center", gap: 6, padding: "3px 6px", borderRadius: 6, flexWrap: "wrap",
            background: lit || row.miss || row.evicted ? `${c}14` : "transparent", border: `1px solid ${lit || row.miss || row.evicted ? `${c}66` : "transparent"}`,
            transition: "all .25s",
          }}>
            <HashChip h={row.h} lit={lit || row.miss || row.evicted} color={c} strike={row.evicted} />
            <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>= hash(</span>
            <HashChip h={info.parent} />
            <span style={{ display: "flex", gap: 1 }}>
              {info.toks.map((t, i) => <Tok key={i} t={t} size={16} color={T.soft} state="pending" />)}
            </span>
            <span style={{ fontFamily: mono, fontSize: 10, color: T.dim }}>)</span>
            <span style={{ fontFamily: mono, fontSize: 11, color: row.miss ? "#FF8A8A" : row.evicted ? "#FF8A8A" : T.ink, marginLeft: "auto" }}>
              {row.miss ? "miss" : row.evicted ? "evicted" : `→ p${row.b}`}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- variables

const RAW_KEYS = new Set(["req", "victim"]);
export function py(v, key) {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (Array.isArray(v)) return `[${v.map((x) => py(x)).join(", ")}]`;
  if (typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `${k}: ${py(x)}`).join(", ")}}`;
  if (typeof v === "string") {
    if (/^[0-9a-f]{8}$/.test(v)) return `#${short(v)}`;
    return RAW_KEYS.has(key) || key === "batch" ? v : `'${v}'`;
  }
  return String(v);
}

export function Frames({ ev, prev }) {
  const frames = ev.snap.frames;
  if (!frames.length) return <div style={{ fontFamily: mono, fontSize: 12, color: T.dim }}>(top level)</div>;
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {frames.map((f, d) => {
        const pf = prev && prev.snap.frames[d] && prev.snap.frames[d].fn === f.fn ? prev.snap.frames[d] : null;
        const top = d === frames.length - 1;
        return (
          <div key={d} style={{
            marginLeft: d * 10, padding: "6px 8px", borderRadius: 7, transition: "all .25s",
            border: `1px solid ${top ? `${ev.color}88` : T.rule}`, background: top ? `${ev.color}0c` : T.well,
          }}>
            <div style={{ fontFamily: mono, fontSize: 11, color: top ? ev.color : T.dim, fontWeight: 700, marginBottom: 4 }}>{f.fn}()</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {Object.entries(f.vars).map(([k, v]) => {
                const changed = !pf || JSON.stringify(pf.vars[k]) !== JSON.stringify(v);
                return (
                  <span key={k} className={changed && top ? "var-flash" : ""} style={{
                    fontFamily: mono, fontSize: 11, padding: "2px 6px", borderRadius: 4, "--c": ev.color,
                    border: `1px solid ${changed && top ? ev.color : T.rule}`, background: T.panel, color: T.soft,
                  }}>
                    <span style={{ color: T.dim }}>{k}</span> = <span style={{ color: T.ink }}>{py(v, k)}</span>
                  </span>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- queues

const LANES = [
  { id: "future", label: "not arrived" },
  { id: "waiting", label: "waiting · deque" },
  { id: "running", label: "running" },
  { id: "done", label: "finished" },
];

export function QueueBoard({ ev }) {
  const snap = ev.snap;
  const pos = {};
  const lists = {
    future: snap.reqs.filter((r) => !snap.waiting.includes(r.id) && !snap.running.includes(r.id) && !snap.done.includes(r.id)).map((r) => r.id),
    waiting: snap.waiting, running: snap.running, done: snap.done,
  };
  LANES.forEach((l, c) => lists[l.id].forEach((id, k) => { pos[id] = { c, k }; }));
  const H = 40;
  const rows = Math.max(1, ...Object.values(lists).map((l) => l.length));
  const sb = Object.fromEntries((snap.batch || []).map((s) => [s.rid, s]));
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 8, marginBottom: 6 }}>
        {LANES.map((l) => (
          <div key={l.id} style={{ fontFamily: mono, fontSize: 10, color: T.dim, textTransform: "uppercase", letterSpacing: 1 }}>
            {l.label}
            {l.id === "waiting" && <span style={{ color: T.rule }}> · head ↑</span>}
            {l.id === "running" && <span style={{ color: T.rule }}> · old ↑</span>}
          </div>
        ))}
      </div>
      <div style={{ position: "relative", height: rows * H + 8 }}>
        {LANES.map((l, c) => (
          <div key={l.id} style={{ position: "absolute", top: 0, bottom: 0, left: `calc(${c * 25}% + 2px)`, width: "calc(25% - 8px)", borderRadius: 8, background: T.well, border: `1px solid ${T.rule}` }} />
        ))}
        {snap.reqs.map((r) => {
          const p = pos[r.id];
          if (!p) return null;
          const active = ev.req === r.id;
          const s = sb[r.id];
          const pct = Math.min(100, (100 * r.kvWritten) / r.token_ids.length);
          return (
            <div key={r.id} style={{
              position: "absolute", left: `calc(${p.c * 25}% + 8px)`, top: 4 + p.k * H, width: "calc(25% - 20px)", height: H - 8,
              transition: "left .5s cubic-bezier(.5,0,.2,1), top .5s cubic-bezier(.5,0,.2,1), box-shadow .3s",
              borderRadius: 7, background: T.panel, border: `1px solid ${active ? ev.color : r.color}66`,
              boxShadow: active ? `0 0 0 2px ${ev.color}, 0 0 14px ${ev.color}55` : "none",
              display: "flex", alignItems: "center", gap: 6, padding: "0 7px", overflow: "hidden",
            }}>
              <Pill id={r.id} color={r.color} small />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ height: 4, borderRadius: 2, background: T.rule, overflow: "hidden" }}>
                  <div style={{ height: "100%", width: `${pct}%`, background: r.color, transition: "width .4s" }} />
                </div>
                <div style={{ fontFamily: mono, fontSize: 9, color: T.dim, marginTop: 2, whiteSpace: "nowrap" }}>
                  {r.kvWritten}/{r.token_ids.length}{s && snap.running.includes(r.id) ? ` · +${s.num_new}` : ""}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// token budget: one cell per token, filled by plan[] in schedule order
export function BudgetBar({ ev, budget }) {
  const snap = ev.snap;
  const sf = snap.frames.find((f) => f.fn === "schedule");
  const plan = sf ? sf.vars.plan : Object.fromEntries((snap.batch || []).map((s) => [s.rid, s.num_new]));
  const cm = colorMap(snap);
  const cells = [];
  for (const [rid, n] of Object.entries(plan || {})) for (let k = 0; k < n; k++) cells.push({ rid, first: k === 0 });
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: mono, fontSize: 10, color: T.dim, width: 54, flexShrink: 0 }}>budget</span>
      <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
        {Array.from({ length: budget }, (_, k) => {
          const c = cells[k];
          return (
            <span key={k} style={{
              width: 26, height: 22, borderRadius: 4, display: "inline-flex", alignItems: "center", justifyContent: "center",
              fontFamily: mono, fontSize: 9.5, fontWeight: 700, transition: "all .3s",
              background: c ? `${cm[c.rid]}40` : "transparent", border: `1px solid ${c ? cm[c.rid] : T.rule}`, color: c ? T.ink : T.rule,
            }}>{c && c.first ? c.rid : ""}</span>
          );
        })}
      </div>
      <span style={{ fontFamily: mono, fontSize: 11, color: T.soft, marginLeft: "auto" }}>{budget - cells.length} left</span>
    </div>
  );
}

import { T, mono } from "../theme";
import { EOS } from "../flow/world";

// Side panels: pure views over one event's world snapshot (w) and what the
// current line touched (hl: req, pages, hashes, lookup, hashBlock, evicted, sampled).

const HIT = "#A5D6A7";
const HASH = "#C49CFF";
const PH = "#FFC94D";
const short = (h) => (h ? h.slice(0, 4) : "");
const colorMap = (w) => Object.fromEntries([...w.clients, ...w.reqs].map((r) => [r.id, r.color]));
export const tokText = (t) => (t === -1 ? "?" : t === EOS ? "⏎" : t === null || t === undefined ? "" : String(t));

export function Section({ title, right, children, color }) {
  return (
    <section style={{ background: T.panel, border: `1px solid ${color ? `${color}66` : T.rule}`, borderRadius: 10, padding: "10px 12px", minWidth: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
        <span style={{ fontFamily: mono, fontSize: 10.5, letterSpacing: 1.2, textTransform: "uppercase", color: color || T.dim, fontWeight: 600 }}>{title}</span>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Pill({ id, color, dim, small }) {
  return (
    <span style={{
      fontFamily: mono, fontSize: small ? 10 : 11, fontWeight: 700, padding: small ? "0 5px" : "1px 7px", borderRadius: 999,
      color: dim ? T.dim : T.bg, background: dim ? "transparent" : color, border: `1px solid ${color}`, whiteSpace: "nowrap",
    }}>{id}</span>
  );
}

export function HashChip({ h, lit, color = HASH, strike }) {
  if (h == null) return <span style={{ fontFamily: mono, fontSize: 10, color: T.rule }}>—</span>;
  return (
    <span style={{
      fontFamily: mono, fontSize: 10, padding: "0 5px", borderRadius: 4, whiteSpace: "nowrap",
      color: lit ? T.bg : color, background: lit ? color : `${color}14`, border: `1px solid ${color}${lit ? "" : "55"}`,
      textDecoration: strike ? "line-through" : "none", transition: "all .25s",
    }}>#{short(h)}</span>
  );
}

// one token slot
export function Tok({ t, color = T.soft, state, gen, size = 20, pop }) {
  const ph = t === -1;
  const s = {
    width: size, height: size, borderRadius: 4, display: "inline-flex", alignItems: "center", justifyContent: "center",
    fontFamily: mono, fontSize: size * 0.55, fontWeight: 600, fontStyle: gen ? "italic" : "normal", flexShrink: 0,
    transition: "background .3s, border-color .3s, color .3s, opacity .3s",
  };
  const st = ph ? { background: `${PH}18`, border: `1.5px dashed ${PH}`, color: PH } : {
    kv: { background: `${color}38`, border: `1px solid ${color}`, color: T.ink },
    hit: { background: `${HIT}30`, border: `1px solid ${HIT}`, color: HIT },
    sched: { background: `${color}12`, border: `1.5px dashed ${color}`, color },
    pending: { background: "transparent", border: `1px solid ${T.rule}`, color: T.soft },
    empty: { background: "transparent", border: `1px dotted ${T.rule}`, color: "transparent" },
    stale: { background: "transparent", border: `1px solid ${T.rule}`, color: T.dim, opacity: 0.4, textDecoration: "line-through" },
    plain: { background: T.well, border: `1px solid ${T.rule}`, color: T.soft },
  }[state];
  return <span className={`${state === "sched" ? "tok-pulse" : ""} ${pop ? "tok-pop" : ""}`} style={{ ...s, ...st, "--c": color }}>{tokText(t)}</span>;
}

const Mono = ({ c = T.dim, s = 10.5, children, style }) => <span style={{ fontFamily: mono, fontSize: s, color: c, ...style }}>{children}</span>;

// ---------------------------------------------------------------- requests

export function RequestRows({ w, hl, color, bs }) {
  if (!w.reqs.length) return <Mono>no requests yet</Mono>;
  const sched = [...(w.next_step ? w.next_step.batch : []), ...(w.runner && w.runner.fwd ? w.runner.fwd : [])];
  return (
    <div style={{ display: "grid", gap: 8 }}>
      {w.reqs.map((r) => {
        const active = hl.req === r.id;
        const done = r.finished && !r.pending;
        const sb = sched.filter((s) => s.rid === r.id);
        const nBlocks = Math.max(Math.ceil(r.token_ids.length / bs), r.block_ids.length);
        const state = (p) => {
          if (p < r.kvWritten) return p < r.hit ? "hit" : "kv";
          if (sb.some((s) => p >= s.num_computed && p < s.num_computed + s.num_new)) return "sched";
          return "pending";
        };
        return (
          <div key={r.id} style={{
            display: "flex", gap: 10, alignItems: "flex-start", padding: "6px 8px", borderRadius: 8,
            background: active ? `${color}10` : "transparent", border: `1px solid ${active ? `${color}66` : "transparent"}`,
            opacity: done ? 0.5 : 1, transition: "all .3s",
          }}>
            <div style={{ width: 50, flexShrink: 0, paddingTop: 2, display: "grid", gap: 3, justifyItems: "start" }}>
              <Pill id={r.id} color={r.color} dim={done} />
              <Mono s={9.5}>nc={r.num_computed}</Mono>
              {r.finished && <Mono s={9} c={done ? T.dim : PH}>{r.finish_reason}</Mono>}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {Array.from({ length: nBlocks }, (_, j) => {
                const page = r.block_ids[j];
                const h = r.block_hashes[j];
                const hlit = hl.hashBlock && hl.hashBlock.rid === r.id && hl.hashBlock.i === j;
                const plit = page !== undefined && hl.pages && hl.pages[page] && active;
                return (
                  <div key={j} style={{ display: "grid", gap: 3, justifyItems: "center" }}>
                    <div style={{ display: "flex", gap: 2, padding: 3, borderRadius: 6, border: `1px solid ${plit ? color : T.rule}`, background: T.well, position: "relative" }}>
                      {Array.from({ length: bs }, (_, o) => {
                        const p = j * bs + o;
                        if (p >= r.token_ids.length) return <Tok key={o} state="empty" />;
                        return (
                          <span key={o} style={{ position: "relative", display: "inline-flex" }}>
                            {p === r.num_computed && <Cursor />}
                            <Tok t={r.token_ids[p]} color={r.color} state={state(p)} gen={p >= r.prompt_len} pop={hl.sampled === p && active} />
                          </span>
                        );
                      })}
                      {r.num_computed === (j + 1) * bs && r.num_computed <= r.token_ids.length && <Cursor right />}
                    </div>
                    <div style={{ display: "flex", gap: 3, alignItems: "center", minHeight: 16 }}>
                      <span style={{
                        fontFamily: mono, fontSize: 9.5, padding: "0 4px", borderRadius: 4, transition: "all .25s",
                        color: page === undefined ? T.rule : plit ? T.bg : T.soft, background: plit ? color : page === undefined ? "transparent" : T.panel,
                        border: `1px solid ${page === undefined ? T.rule : plit ? color : T.dim}`,
                      }}>{page === undefined ? "p ?" : `p${page}`}</span>
                      {h && <HashChip h={h} lit={hlit} color={hl.lookup && hlit ? (hl.lookup.hit ? HIT : "#FF8A8A") : HASH} />}
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

function Cursor({ right }) {
  return <span style={{ position: "absolute", [right ? "right" : "left"]: right ? 0 : -3, top: -3, bottom: -3, width: 2, borderRadius: 1, background: T.accent, boxShadow: `0 0 6px ${T.accent}` }} />;
}

export function TokenLegend() {
  const item = (label, style) => (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
      <span style={{ width: 11, height: 11, borderRadius: 3, ...style }} />
      <Mono s={9.5}>{label}</Mono>
    </span>
  );
  return (
    <span style={{ display: "inline-flex", gap: 9, flexWrap: "wrap" }}>
      {item("K/V written", { background: "#5FD3F338", border: "1px solid #5FD3F3" })}
      {item("from cache", { background: "#A5D6A730", border: "1px solid #A5D6A7" })}
      {item("scheduled", { border: "1.5px dashed #5FD3F3" })}
      {item("? placeholder", { border: `1.5px dashed ${PH}` })}
      {item("num_computed", { width: 3, background: T.accent })}
    </span>
  );
}

// ---------------------------------------------------------------- queues + budget

const LANES = [
  { id: "future", label: "not sent" },
  { id: "waiting", label: "waiting ↑head" },
  { id: "running", label: "running ↑old" },
  { id: "done", label: "finished" },
];

export function QueueBoard({ w, hl, color }) {
  const pos = {};
  const known = new Set(w.reqs.map((r) => r.id));
  const lists = { future: w.clients.filter((c) => !known.has(c.id)).map((c) => c.id), waiting: w.waiting, running: w.running, done: w.done };
  LANES.forEach((l, c) => lists[l.id].forEach((id, k) => { pos[id] = { c, k }; }));
  const H = 30;
  const rows = Math.max(1, ...Object.values(lists).map((l) => l.length));
  const cm = colorMap(w);
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 6, marginBottom: 4 }}>
        {LANES.map((l) => <Mono key={l.id} s={9.5} style={{ textTransform: "uppercase", letterSpacing: 0.8 }}>{l.label}</Mono>)}
      </div>
      <div style={{ position: "relative", height: rows * H + 6 }}>
        {LANES.map((l, c) => (
          <div key={l.id} style={{ position: "absolute", top: 0, bottom: 0, left: `calc(${c * 25}% + 1px)`, width: "calc(25% - 6px)", borderRadius: 7, background: T.well, border: `1px solid ${T.rule}` }} />
        ))}
        {[...w.clients].map((c) => {
          const p = pos[c.id];
          if (!p) return null;
          const r = w.reqs.find((x) => x.id === c.id);
          const active = hl.req === c.id;
          const pct = r ? Math.min(100, (100 * r.kvWritten) / r.token_ids.length) : 0;
          return (
            <div key={c.id} style={{
              position: "absolute", left: `calc(${p.c * 25}% + 5px)`, top: 3 + p.k * H, width: "calc(25% - 14px)", height: H - 6,
              transition: "left .5s cubic-bezier(.5,0,.2,1), top .5s cubic-bezier(.5,0,.2,1), box-shadow .3s",
              borderRadius: 6, background: T.panel, border: `1px solid ${cm[c.id]}66`,
              boxShadow: active ? `0 0 0 2px ${color}, 0 0 12px ${color}55` : "none",
              display: "flex", alignItems: "center", gap: 5, padding: "0 5px", overflow: "hidden",
            }}>
              <Pill id={c.id} color={cm[c.id]} small />
              <div style={{ flex: 1, minWidth: 0, height: 4, borderRadius: 2, background: T.rule, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${pct}%`, background: cm[c.id], transition: "width .4s" }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function BudgetBar({ w, plan, budget }) {
  const cm = colorMap(w);
  const cells = [];
  for (const [rid, n] of Object.entries(plan || {})) for (let k = 0; k < n; k++) cells.push({ rid, first: k === 0 });
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <Mono s={9.5} style={{ width: 44, flexShrink: 0 }}>budget</Mono>
      <div style={{ display: "flex", gap: 3, flexWrap: "wrap" }}>
        {Array.from({ length: budget }, (_, k) => {
          const c = cells[k];
          return (
            <span key={k} style={{
              width: 24, height: 20, borderRadius: 4, display: "inline-flex", alignItems: "center", justifyContent: "center",
              fontFamily: mono, fontSize: 9, fontWeight: 700, transition: "all .3s",
              background: c ? `${cm[c.rid]}40` : "transparent", border: `1px solid ${c ? cm[c.rid] : T.rule}`, color: c ? T.ink : T.rule,
            }}>{c && c.first ? c.rid : ""}</span>
          );
        })}
      </div>
      <Mono s={10.5} c={T.soft} style={{ marginLeft: "auto" }}>{budget - cells.length} left</Mono>
    </div>
  );
}

// ---------------------------------------------------------------- pages

export function PagePool({ w, hl, color }) {
  if (!w.kv) return <Mono>KVCacheManager not built yet</Mono>;
  const cm = colorMap(w);
  const owners = (b) => w.reqs.filter((r) => r.block_ids.includes(b));
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      {w.pages.map((slots, b) => {
        const own = owners(b);
        const lru = w.kv.free.indexOf(b);
        const act = hl.pages && hl.pages[b];
        const isNull = b === 0;
        const border = act ? color : isNull ? T.rule : own.length ? own[0].color : T.dim;
        return (
          <div key={b} style={{
            position: "relative", padding: "5px 5px 4px", borderRadius: 7, minWidth: 64,
            border: `${act ? 2 : 1}px ${!isNull && !own.length ? "dashed" : "solid"} ${border}`,
            background: isNull ? `repeating-linear-gradient(45deg, ${T.well}, ${T.well} 5px, ${T.panel} 5px, ${T.panel} 10px)` : T.well,
            boxShadow: act ? `0 0 12px ${color}55` : "none", transition: "all .3s", margin: act ? 0 : 1,
          }}>
            {act && <span style={{ position: "absolute", top: -8, right: 4, fontFamily: mono, fontSize: 9, fontWeight: 700, padding: "0 4px", borderRadius: 4, background: color, color: T.bg }}>{act}</span>}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 5, marginBottom: 4 }}>
              <Mono s={10.5} c={isNull ? T.dim : T.ink} style={{ fontWeight: 700 }}>p{b}</Mono>
              {isNull ? <Mono s={9}>null</Mono> : w.kv.ref[b] > 0 ? <Mono s={9} c={border}>ref {w.kv.ref[b]}</Mono> : <Mono s={9}>LRU {lru + 1}</Mono>}
            </div>
            <div style={{ display: "flex", gap: 2 }}>
              {slots.map((s, o) => <Tok key={o} size={16} t={s ? s.tok : null} color={s ? cm[s.rid] || T.dim : T.rule} state={!s ? "empty" : s.stale ? "stale" : "kv"} gen={s && s.gen} />)}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4, gap: 4, minHeight: 15 }}>
              <HashChip h={w.kv.bh[b]} lit={act === "hash"} />
              <span style={{ display: "flex", gap: 2 }}>{own.map((r) => <span key={r.id} title={r.id} style={{ width: 7, height: 7, borderRadius: 4, background: r.color }} />)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function FreeList({ w, hl, color }) {
  if (!w.kv) return null;
  const W = 42;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
      <Mono s={9.5} style={{ width: 44, flexShrink: 0, lineHeight: 1.25 }}>LRU<br />front</Mono>
      <div style={{ position: "relative", height: 30, flex: 1, minWidth: 0, borderRadius: 6, background: T.well, border: `1px solid ${T.rule}`, overflow: "hidden" }}>
        {w.kv.free.map((b, k) => {
          const act = hl.pages && hl.pages[b];
          const cached = w.kv.bh[b] != null;
          return (
            <span key={b} style={{
              position: "absolute", top: 3, left: 3 + k * W, width: W - 6, height: 22, borderRadius: 5,
              display: "flex", alignItems: "center", justifyContent: "center", gap: 3,
              fontFamily: mono, fontSize: 10.5, fontWeight: 600, transition: "left .45s ease, background .3s",
              color: act ? T.bg : T.soft, background: act ? color : T.panel, border: `1px solid ${cached ? HASH : T.rule}`,
            }}>p{b}{cached && <span style={{ width: 5, height: 5, borderRadius: 3, background: act ? T.bg : HASH }} />}</span>
          );
        })}
        {!w.kv.free.length && <Mono s={10.5} c="#FF8A8A" style={{ position: "absolute", left: 8, top: 7 }}>empty</Mono>}
      </div>
      <Mono s={9.5} style={{ width: 34, flexShrink: 0, lineHeight: 1.25 }}>tail<br />← free</Mono>
    </div>
  );
}

export function HashTable({ w, hl, hashInfo }) {
  if (!w.kv) return null;
  const rows = w.kv.h2b.map(([h, b]) => ({ h, b }));
  const extra = [];
  if (hl.lookup && !hl.lookup.hit) extra.push({ h: hl.lookup.h, miss: true });
  for (const h of hl.evicted || []) extra.push({ h, evicted: true });
  if (!rows.length && !extra.length) return <Mono>{"{}"}</Mono>;
  return (
    <div style={{ display: "grid", gap: 3 }}>
      {[...rows, ...extra].map((row) => {
        const info = hashInfo[row.h] || { parent: null, toks: [] };
        const lit = (hl.hashes && hl.hashes.includes(row.h)) || (hl.lookup && hl.lookup.h === row.h);
        const bad = row.miss || row.evicted;
        const c = bad ? "#FF8A8A" : hl.lookup && hl.lookup.h === row.h ? HIT : HASH;
        return (
          <div key={row.h + (row.miss ? "m" : row.evicted ? "e" : "")} style={{
            display: "flex", alignItems: "center", gap: 5, padding: "2px 5px", borderRadius: 6, flexWrap: "wrap",
            background: lit || bad ? `${c}14` : "transparent", border: `1px solid ${lit || bad ? `${c}66` : "transparent"}`, transition: "all .25s",
          }}>
            <HashChip h={row.h} lit={lit || bad} color={c} strike={row.evicted} />
            <Mono s={9.5}>= hash(</Mono>
            <HashChip h={info.parent} />
            <span style={{ display: "flex", gap: 1 }}>{info.toks.map((t, i) => <Tok key={i} t={t} size={14} state="pending" />)}</span>
            <Mono s={9.5}>)</Mono>
            <Mono s={10.5} c={bad ? "#FF8A8A" : T.ink} style={{ marginLeft: "auto" }}>{row.miss ? "miss" : row.evicted ? "evicted" : `→ p${row.b}`}</Mono>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- async pipeline (Engine.step)

const SRChips = ({ batch, cm }) => (
  <span style={{ display: "inline-flex", gap: 4, flexWrap: "wrap" }}>
    {batch.map((s, i) => (
      <span key={i} style={{ fontFamily: mono, fontSize: 10, padding: "1px 6px", borderRadius: 5, border: `1px solid ${cm[s.rid]}`, background: `${cm[s.rid]}22`, color: T.ink }}>
        {s.rid} <span style={{ color: T.dim }}>{s.num_computed}+{s.num_new}</span>
      </span>
    ))}
  </span>
);

const TokArr = ({ label, arr, pending, n, sub }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
    <Mono s={9.5} style={{ width: 92, flexShrink: 0 }}>{label}<br /><span style={{ color: T.rule }}>{sub}</span></Mono>
    <span style={{ display: "inline-flex", gap: 2, opacity: pending ? 0.55 : 1 }}>
      {arr.map((t, i) => <Tok key={i} t={i < n ? t : null} size={18} state={i < n ? (pending ? "sched" : "plain") : "empty"} color={PH} />)}
    </span>
    {pending && <Mono s={9.5} c={PH}>copy in flight…</Mono>}
  </div>
);

export function Pipeline({ w }) {
  if (!w.runner) return <Mono>ModelRunner not built yet</Mono>;
  const cm = colorMap(w);
  const nIn = w.in_flight ? w.in_flight.batch.length : 0;
  return (
    <div style={{ display: "grid", gap: 7 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Mono s={9.5} style={{ width: 92 }}>next_step<br /><span style={{ color: T.rule }}>packed, not run</span></Mono>
        {w.next_step ? <><SRChips batch={w.next_step.batch} cm={cm} /><Mono s={10} c="#C49CFF">{w.next_step.mode}{w.next_step.size ? `:${w.next_step.size}` : ""}</Mono></> : <Mono>None</Mono>}
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Mono s={9.5} style={{ width: 92 }}>in_flight<br /><span style={{ color: T.rule }}>ran, tokens due</span></Mono>
        {w.in_flight ? <><SRChips batch={w.in_flight.batch} cm={cm} /><Mono s={10}>live [{w.in_flight.live.join(", ")}]</Mono></> : <Mono>None</Mono>}
      </div>
      <TokArr label="prev_tokens" sub="GPU" arr={w.runner.prev_tokens} n={Math.max(nIn, w.runner.prev_tokens.filter((t) => t !== 0).length)} />
      <TokArr label="tokens_host" sub="pinned CPU" arr={w.runner.tokens_host} pending={w.runner.hostPending} n={Math.max(nIn, w.runner.tokens_host.filter((t) => t !== 0).length)} />
    </div>
  );
}

// ---------------------------------------------------------------- packed batch (_pack / _upload)

export function Packed({ w, frames }) {
  const pf = frames.find((f) => f.fn === "ModelRunner._pack");
  const cm = colorMap(w);
  let rows, qsl, seq, bt, logit, fill;
  if (pf && pf.vars.ids) {
    const v = pf.vars;
    const batch = v.batch || [];
    qsl = v.qsl || [0];
    const owner = (r) => { for (let i = 0; i + 1 < qsl.length; i++) if (r >= qsl[i] && r < qsl[i + 1]) return batch[i] ? batch[i].$sr[0] : null; return null; };
    rows = v.ids.map((t, r) => ({ t, pos: (v.pos || [])[r], slot: (v.slots || [])[r], rid: owner(r) }));
    seq = v.seq_lens || [];
    bt = v.bt || [];
    logit = v.logit_rows || [];
    fill = (v.fill_rows || []).map((r, k) => [r, v.fill_src[k]]);
  } else if (w.runner && w.runner.buf) {
    const b = w.runner.buf;
    rows = b.input_ids.map((t, r) => ({ t, pos: b.positions[r], slot: b.slot_mapping[r], rid: b.owner[r] }));
    qsl = b.qsl; seq = b.seq_lens; bt = b.bt; logit = b.logit_rows; fill = b.fill;
  } else return <Mono>nothing packed yet</Mono>;
  const bs = w.pages[0] ? w.pages[0].length : 4;
  const cell = { fontFamily: mono, fontSize: 10.5, padding: "1px 5px", textAlign: "center", borderBottom: `1px solid ${T.rule}` };
  const head = { ...cell, color: T.dim, textAlign: "right", paddingRight: 8 };
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse" }}>
          <tbody>
            <tr><td style={head}>row</td>{rows.map((_, r) => <td key={r} style={{ ...cell, color: T.dim }}>{r}</td>)}</tr>
            <tr><td style={head}>ids</td>{rows.map((x, r) => {
              const f = fill.find((p) => p[0] === r);
              return <td key={r} style={cell}><Tok t={x.t === 0 && !x.rid ? 0 : x.t} size={18} color={cm[x.rid] || T.dim} state={x.rid ? "kv" : "plain"} />{f && <div style={{ fontSize: 8.5, color: PH }}>←prev[{f[1]}]</div>}</td>;
            })}</tr>
            <tr><td style={head}>pos</td>{rows.map((x, r) => <td key={r} style={cell}>{x.pos ?? ""}</td>)}</tr>
            <tr><td style={head}>slot</td>{rows.map((x, r) => <td key={r} style={{ ...cell, color: x.rid ? T.ink : T.dim }}>{x.slot === undefined ? "" : `${Math.floor(x.slot / bs)}:${x.slot % bs}`}</td>)}</tr>
            <tr><td style={head}>sample</td>{rows.map((_, r) => <td key={r} style={{ ...cell, color: T.accent }}>{logit.includes(r) ? "★" : ""}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <Mono s={10.5} c={T.soft}>qsl = [{qsl.join(", ")}] · seq_lens = [{seq.join(", ")}] · slot = page:offset</Mono>
      {bt.length > 0 && (
        <div style={{ display: "grid", gap: 2 }}>
          <Mono s={9.5}>block_table</Mono>
          {bt.map((row, i) => (
            <span key={i} style={{ display: "inline-flex", gap: 2 }}>
              {row.map((b, k) => <span key={k} style={{ fontFamily: mono, fontSize: 10, width: 20, textAlign: "center", borderRadius: 3, border: `1px solid ${b ? T.dim : T.rule}`, color: b ? T.ink : T.rule }}>{b}</span>)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- model forward

export function ModelView({ w, layers }) {
  const cm = colorMap(w);
  const a = w.attn;
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        <Mono s={9.5} style={{ width: 44 }}>layer</Mono>
        {Array.from({ length: layers }, (_, k) => (
          <span key={k} style={{ fontFamily: mono, fontSize: 10.5, padding: "2px 10px", borderRadius: 5, transition: "all .3s",
            border: `1px solid ${w.layer === k ? "#8AB4FF" : T.rule}`, background: w.layer === k ? "#8AB4FF33" : "transparent", color: w.layer === k ? T.ink : T.dim }}>{k}</span>
        ))}
        <Mono s={9.5} style={{ marginLeft: 8 }}>each layer has its own K/V pool, same page ids</Mono>
      </div>
      {a && a.layer === w.layer && a.reqs.map((q, i) => (
        <div key={i} style={{ display: "grid", gap: 3 }}>
          <Mono s={10} c={q.rid ? T.soft : T.dim}>{q.rid ? `${q.rid}: ${q.qtoks.length} queries × ${q.kv} keys (pages ${q.pages.join(", ")})` : "padding row → null page"}</Mono>
          {q.rid && (
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "separate", borderSpacing: 2 }}>
                <tbody>
                  <tr><td />{q.ktoks.map((t, k) => <td key={k}><Tok t={t} size={16} color={cm[q.rid]} state="plain" /></td>)}</tr>
                  {q.qtoks.map((t, r) => (
                    <tr key={r}>
                      <td><Tok t={t} size={16} color={cm[q.rid]} state="kv" /></td>
                      {q.ktoks.map((_, k) => <td key={k}><span style={{ display: "block", width: 16, height: 16, borderRadius: 3, background: k <= q.q0 + r ? `${cm[q.rid]}77` : T.well, border: `1px solid ${T.rule}` }} /></td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ))}
      {(!a || a.layer !== w.layer) && <Mono>attention runs at the end of Attention.attend</Mono>}
    </div>
  );
}

// ---------------------------------------------------------------- sampling

export function SampleView({ w, frames }) {
  const cm = colorMap(w);
  const ls = frames.find((f) => f.fn === "ModelRunner.launch_sample");
  const rp = frames.find((f) => f.fn === "apply_repetition_penalty");
  const batch = ls && ls.vars.batch ? ls.vars.batch : [];
  const toks = ls && ls.vars.tokens;
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {batch.map((s, i) => {
        const [rid, nc, nn] = s.$sr;
        const r = w.reqs.find((x) => x.id === rid);
        const mid = r && nc + nn < r.token_ids.length - (r.pending ? 1 : 0);
        return (
          <div key={i} style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Mono s={10} style={{ width: 40 }}>row {i}</Mono>
            <Pill id={rid} color={cm[rid]} small />
            <Mono s={10}>logits[{i}] → argmax</Mono>
            {toks ? <Tok t={toks[i]} size={20} color={cm[rid]} state={toks[i] === "~" ? "stale" : "kv"} pop /> : <Mono>…</Mono>}
            {toks && toks[i] === "~" && <Mono s={9.5} c={PH}>mid-prompt chunk: dropped</Mono>}
            {mid && !toks && <Mono s={9.5} c={PH}>mid-prompt</Mono>}
          </div>
        );
      })}
      {rp && rp.vars.windows && (
        <div style={{ display: "grid", gap: 3, marginTop: 4 }}>
          <Mono s={9.5}>repetition-penalty windows (last repeat_last_n ids)</Mono>
          {rp.vars.windows.map((win, k) => (
            <span key={k} style={{ display: "inline-flex", gap: 1, flexWrap: "wrap" }}>{win.map((t, j) => <Tok key={j} t={t} size={14} state="pending" />)}</span>
          ))}
        </div>
      )}
      {!batch.length && <Mono>no batch</Mono>}
    </div>
  );
}

// ---------------------------------------------------------------- outputs / HTTP

export function Outputs({ w }) {
  const cm = colorMap(w);
  const ids = Object.keys(w.outputs);
  if (!ids.length) return <Mono>no output yet</Mono>;
  return (
    <div style={{ display: "grid", gap: 4 }}>
      {ids.map((id) => {
        const r = w.reqs.find((x) => x.id === id);
        const st = w.async && w.async.streams[id];
        return (
          <div key={id} style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Pill id={id} color={cm[id]} small />
            <Mono s={12} c={T.ink}>'{w.outputs[id]}'</Mono>
            {r && r.finished && <Mono s={9.5}>{r.finish_reason}</Mono>}
            {st && <Mono s={9.5} c="#5FD3F3" style={{ marginLeft: "auto" }}>client got '{st.received}'{st.queued ? ` +${st.queued} queued` : ""}</Mono>}
          </div>
        );
      })}
    </div>
  );
}

export function AsyncView({ w }) {
  if (!w.async) return <Mono>AsyncEngine not built yet</Mono>;
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <Mono s={9.5} style={{ width: 70 }}>inbox</Mono>
        {w.async.inbox.length ? w.async.inbox.map((m, i) => <span key={i} style={{ fontFamily: mono, fontSize: 10.5, padding: "1px 6px", borderRadius: 5, border: "1px solid #5FD3F3", color: T.ink }}>{m}</span>) : <Mono>empty</Mono>}
      </div>
      <Outputs w={w} />
    </div>
  );
}

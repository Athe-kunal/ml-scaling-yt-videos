import { T, mono } from "../theme";
import { AsyncView, BudgetBar, FreeList, HashTable, ModelView, Outputs, Packed, PagePool, Pipeline, QueueBoard, RequestRows, SampleView } from "./Panels";

// The picture drawn inside the running block: the part of the world that block
// changes, at the moment of the current event.

function planOf(w, frames) {
  const sched = frames.find((f) => f.fn === "Scheduler.schedule");
  if (sched && sched.vars.plan) return sched.vars.plan;
  return Object.fromEntries((w.next_step ? w.next_step.batch : []).map((s) => [s.rid, s.num_new]));
}

function Config({ cfg, w }) {
  const chip = (k, v) => (
    <span key={k} style={{ fontFamily: mono, fontSize: 10.5, padding: "1px 7px", borderRadius: 5, border: `1px solid ${T.rule}`, background: T.bg, color: T.soft }}>
      <span style={{ color: T.dim }}>{k}</span> {v}
    </span>
  );
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {chip("block_size", cfg.bs)}
      {chip("num_blocks", cfg.numBlocks)}
      {chip("batched_tokens", cfg.budget)}
      {chip("max_seqs", cfg.maxSeqs)}
      {chip("layers", cfg.layers)}
      {w.runner && chip("graphs", w.runner.graphs.join(" ") || "none yet")}
    </div>
  );
}

// the request the innermost frames are working on
function curReq(frames, hl) {
  for (let d = frames.length - 1; d >= 0; d--) {
    const r = frames[d].vars.req;
    if (r && r.$r) return r.$r;
  }
  return hl.req || null;
}

export function BlockViz({ kind, ev, frames, cfg, hashInfo, color }) {
  const w = ev.world, hl = ev.hl;
  const rid = curReq(frames, hl);
  const one = rid ? { ...w, reqs: w.reqs.filter((r) => r.id === rid) } : w;
  const body = {
    config: <Config cfg={cfg} w={w} />,
    requests: <RequestRows w={w} hl={hl} color={color} bs={cfg.bs} />,
    queues: <QueueBoard w={w} hl={hl} color={color} />,
    sched: <div style={{ display: "grid", gap: 8 }}><QueueBoard w={w} hl={hl} color={color} /><BudgetBar w={w} plan={planOf(w, frames)} budget={cfg.budget} /></div>,
    budget: <BudgetBar w={w} plan={planOf(w, frames)} budget={cfg.budget} />,
    pages: <div style={{ display: "grid", gap: 8 }}>{rid && <RequestRows w={one} hl={hl} color={color} bs={cfg.bs} />}<PagePool w={w} hl={hl} color={color} /><FreeList w={w} hl={hl} color={color} /></div>,
    hash: <div style={{ display: "grid", gap: 8 }}><RequestRows w={one} hl={hl} color={color} bs={cfg.bs} /><HashTable w={w} hl={hl} hashInfo={hashInfo} /></div>,
    pipeline: <Pipeline w={w} />,
    batch: <Packed w={w} frames={frames} />,
    model: <ModelView w={w} layers={cfg.layers} />,
    sample: <SampleView w={w} frames={frames} />,
    outputs: <Outputs w={w} />,
    inbox: <AsyncView w={w} />,
  }[kind];
  if (!body) return null;
  return (
    <div className="viz-in" style={{ background: T.bg, border: `1px solid ${color}44`, borderRadius: 8, padding: "8px 10px", minWidth: 0 }}>
      {body}
    </div>
  );
}

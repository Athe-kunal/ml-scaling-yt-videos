import { KV } from "./kv.js";
import { call, emit, enter, leave, makeWorld, newReq, writeKV } from "./trace.js";

// Port of tinyserve.Scheduler + the parts of Engine.step that touch it.
// The model is faked: each request has a fixed list of output tokens.

class Scheduler {
  constructor(w, cfg, kv) {
    this.w = w;
    this.cfg = cfg;
    this.kv = kv;
  }

  schedule() {
    const w = this.w, kv = this.kv, cfg = this.cfg, bs = w.bs;
    const f = enter(w, "schedule", {});
    let budget = cfg.budget;
    const plan = {};
    let preempted = false;
    let i = 0;
    const sync = () => Object.assign(f.vars, { budget, plan: { ...plan }, preempted, i });
    sync();
    emit(w, { code: "sched", tags: ["sc_init"], kind: "p1", notes: [{ tag: "sc_init", t: `budget = ${budget}` }] });

    while (i < w.running.length && budget > 0) {
      const req = w.running[i];
      const n = Math.min(req.token_ids.length - req.num_computed, budget);
      Object.assign(f.vars, { req: req.id, n });
      delete f.vars.cached;
      sync();
      emit(w, {
        code: "sched", tags: ["p1_loop", "p1_req"], kind: "p1", req: req.id,
        notes: [{ tag: "p1_n", t: `${req.id}: min(${req.token_ids.length} - ${req.num_computed}, ${budget}) = ${n}` }],
      });
      while (!call(w, "sched", "p1_alloc", () => kv.allocate(req, n), `allocate(${req.id}, ${n})`)) {
        const victim = w.running.pop();
        f.vars.victim = victim.id;
        emit(w, { code: "sched", tags: ["pre"], kind: "preempt", req: victim.id, notes: [{ tag: "pre", t: `victim = ${victim.id} (newest running)` }] });
        call(w, "sched", "pre_free", () => kv.freeRequest(victim));
        victim.num_computed = 0;
        victim.kvWritten = 0;
        victim.hit = 0;
        w.waiting.unshift(victim);
        budget += plan[victim.id] ?? 0;
        delete plan[victim.id];
        preempted = true;
        sync();
        emit(w, {
          code: "sched", tags: ["pre_after"], kind: "preempt", req: victim.id,
          notes: [{ tag: "pre_after", t: `${victim.id}: computed = 0 → front of waiting` }],
        });
        if (victim === req) {
          emit(w, { code: "sched", tags: ["pre_self"], kind: "preempt", req: req.id, notes: [{ tag: "pre_self", t: "preempted itself" }] });
          break;
        }
      }
      if (!w.running.includes(req)) {
        emit(w, { code: "sched", tags: ["p1_gone"], kind: "preempt", req: req.id, notes: [{ tag: "p1_gone", t: "stop phase 1" }] });
        break;
      }
      plan[req.id] = n;
      budget -= n;
      i += 1;
      sync();
      emit(w, {
        code: "sched", tags: ["p1_plan"], kind: "p1", req: req.id,
        notes: [{ tag: "p1_plan", t: `plan[${req.id}] = ${n} · budget → ${budget}` }],
      });
    }

    let failed = false;
    while (!preempted && w.waiting.length && budget > 0 && w.running.length < cfg.maxSeqs) {
      const req = w.waiting[0];
      f.vars.req = req.id;
      delete f.vars.n;
      delete f.vars.victim;
      sync();
      emit(w, { code: "sched", tags: ["p2_loop", "p2_req"], kind: "p2", req: req.id, notes: [{ tag: "p2_req", t: `head = ${req.id}` }] });
      const cached = call(w, "sched", "p2_cache", () => kv.findCachedPrefix(req));
      const n = Math.min(req.token_ids.length - cached.length * bs, budget);
      Object.assign(f.vars, { cached, n });
      emit(w, {
        code: "sched", tags: ["p2_n"], kind: "p2", req: req.id,
        notes: [{ tag: "p2_n", t: `min(${req.token_ids.length} - ${cached.length}·${bs}, ${budget}) = ${n}` }],
      });
      if (!call(w, "sched", "p2_alloc", () => kv.allocate(req, n, cached), `allocate(${req.id}, ${n}, [${cached.join(", ")}])`)) {
        emit(w, { code: "sched", tags: ["p2_fail"], kind: "fail", req: req.id, notes: [{ tag: "p2_fail", t: "head does not fit → stop admitting (FIFO)" }] });
        failed = true;
        break;
      }
      w.waiting.shift();
      req.num_computed = cached.length * bs;
      req.kvWritten = req.num_computed;
      req.hit = req.num_computed;
      req.num_cached = Math.max(req.num_cached, req.num_computed);
      w.running.push(req);
      plan[req.id] = n;
      budget -= n;
      sync();
      emit(w, {
        code: "sched", tags: ["p2_admit"], kind: "p2", req: req.id,
        notes: [{ tag: "p2_admit", t: `${req.id} → running · computed = ${req.num_computed} · plan ${n}` }],
      });
    }
    if (!failed) {
      const why = preempted ? "preempted → skip phase 2"
        : !w.waiting.length ? "waiting is empty"
          : budget <= 0 ? "budget used up"
            : "max_num_seqs reached";
      emit(w, { code: "sched", tags: ["p2_loop"], kind: "p2", notes: [{ tag: "p2_loop", t: why }] });
    }

    const batch = [];
    for (const req of w.running) {
      if (req.id in plan) {
        batch.push({ rid: req.id, num_computed: req.num_computed, num_new: plan[req.id] });
        req.num_computed += plan[req.id];
      }
    }
    w.batch = batch;
    f.vars.batch = batch.map((s) => `${s.rid}:${s.num_computed}+${s.num_new}`);
    emit(w, {
      code: "sched", tags: ["emit"], kind: "emit",
      notes: [{ tag: "emit", t: batch.map((s) => `${s.rid}(${s.num_computed}+${s.num_new})`).join("  ") || "empty" }],
    });
    leave(w);
    return batch;
  }

  finish(req, reason) {
    const w = this.w;
    enter(w, "finish", { req: req.id, reason });
    req.finish_reason = reason;
    w.running = w.running.filter((r) => r !== req);
    w.waiting = w.waiting.filter((r) => r !== req);
    w.done.push(req);
    emit(w, { code: "sched", tags: ["fin"], kind: "finish", req: req.id, notes: [{ tag: "fin", t: `${req.id} finished (${reason})` }] });
    call(w, "sched", "fin_free", () => this.kv.freeRequest(req));
    leave(w);
  }
}

export const DEFAULT_SIM = { bs: 4, numBlocks: 8, budget: 6, maxSeqs: 3, maxModelLen: 32 };

// Shared prefix "ABCDEFGH" = two full blocks at block size 4.
export const SIM_REQUESTS = [
  { id: "R1", prompt: "ABCDEFGHIJ", outputs: "pqrst", arrival: 0 },
  { id: "R2", prompt: "ABCDEFGHKL", outputs: "stuv", arrival: 1 },
  { id: "R3", prompt: "MNOPQRSTU", outputs: "uvwxy", arrival: 1 },
  { id: "R4", prompt: "ABCDEFGHIJY", outputs: "yz", arrival: 8 },
];

const CAP = 60;

export function runSim(cfg) {
  const w = makeWorld(cfg.bs);
  const kv = new KV(w, cfg.numBlocks, cfg.bs);
  const sch = new Scheduler(w, cfg, kv);
  const reqs = SIM_REQUESTS.map((r) =>
    newReq(w, r.id, [...r.prompt], { arrival: r.arrival, outputs: [...r.outputs], maxTokens: r.outputs.length }));
  emit(w, { code: "sched", tags: [], kind: "init", title: `${cfg.numBlocks} pages × ${cfg.bs} slots · page 0 = null` });

  let stalled = false;
  for (let step = 0; step < CAP; step++) {
    const arriving = reqs.filter((r) => r.arrival === step);
    if (!arriving.length && !w.waiting.length && !w.running.length) {
      if (reqs.every((r) => r.arrival < step)) break;
      continue;
    }
    w.step = step;
    w.group = step;
    for (const r of arriving) {
      w.waiting.push(r);
      emit(w, { code: "sched", tags: ["add"], kind: "arrive", req: r.id, notes: [{ tag: "add", t: `${r.id} → waiting (${r.prompt_len} tokens)` }] });
    }
    const f = enter(w, "step", {});
    const batch = call(w, "sched", "st_sched", () => sch.schedule());
    if (!batch.length) {
      emit(w, { code: "sched", tags: ["st_empty"], kind: "fail", notes: [{ tag: "st_empty", t: "nothing fits this step" }] });
      leave(w);
      if (!w.running.length && w.waiting.length) {
        stalled = true;
        break;
      }
      continue;
    }

    const pages = {};
    for (const s of batch) {
      const req = reqs.find((r) => r.id === s.rid);
      Object.assign(pages, writeKV(w, req, s.num_computed, s.num_computed + s.num_new));
    }
    const T = batch.reduce((a, s) => a + s.num_new, 0);
    emit(w, { code: "sched", tags: ["st_run"], kind: "run", pages, notes: [{ tag: "st_run", t: `forward: ${T} packed tokens → K/V written` }] });

    for (const s of batch) {
      const req = reqs.find((r) => r.id === s.rid);
      f.vars.req = req.id;
      const completes = s.num_computed + s.num_new >= req.token_ids.length;
      if (!completes) {
        emit(w, {
          code: "sched", tags: ["st_loop", "st_skip"], kind: "run", req: req.id,
          notes: [{ tag: "st_skip", t: `${s.num_computed}+${s.num_new} < ${req.token_ids.length}: mid-prompt, sample dropped` }],
        });
        continue;
      }
      const tok = req.outputs[req.token_ids.length - req.prompt_len];
      req.token_ids.push(tok);
      emit(w, { code: "sched", tags: ["st_loop", "st_append"], kind: "run", req: req.id, sampled: req.token_ids.length - 1, notes: [{ tag: "st_append", t: `${req.id} sampled '${tok}'` }] });
      const nOut = req.token_ids.length - req.prompt_len;
      if (nOut >= req.maxTokens || req.token_ids.length >= cfg.maxModelLen) {
        call(w, "sched", "st_len", () => sch.finish(req, "length"));
      }
    }
    leave(w);
  }
  if (stalled) emit(w, { code: "sched", tags: [], kind: "fail", title: "stalled: the cache cannot hold the head request" });
  return { events: w.events, hashInfo: w.hashInfo };
}

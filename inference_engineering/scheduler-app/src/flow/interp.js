import { FNS } from "./dsl.js";
import { snapshotWorld } from "./world.js";

// Runs the flowcharts as JS generators, one generator per Python thread, and
// records an event (a full snapshot) every time a node runs. A thread that
// blocks (inbox.get, await q.get) yields SUSPEND and the driver picks another.

const BREAK = { brk: 1 };
const CONTINUE = { cont: 1 };
class Ret {
  constructor(v) {
    this.v = v;
  }
}
const SUSPEND = { suspend: 1 };
const STEP = { step: 1 };

// ---------------------------------------------------------------- values

// python-ish text for notes
export function fmt(v, depth = 0) {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "string") return v === "-1" ? "?" : `'${v}'`;
  if (typeof v === "number") return v === -1 ? "?" : String(v);
  if (v._req) return v.id;
  if (v._sr) return `SR(${v.req.id}, ${v.num_computed}, ${v.num_new})`;
  if (v._t) return `Tensor[${v._t.join(", ")}]`;
  if (v._x) return v._x;
  if (depth > 1) return "…";
  if (Array.isArray(v)) {
    const xs = v.slice(0, 10).map((x) => fmt(x, depth + 1));
    return `[${xs.join(", ")}${v.length > 10 ? ", …" : ""}]`;
  }
  if (typeof v === "object") return `{${Object.entries(v).slice(0, 6).map(([k, x]) => `${k}: ${fmt(x, depth + 1)}`).join(", ")}}`;
  return String(v);
}

// plain data for the diagram's variable chips
function ser(v, depth = 0) {
  if (v === null || v === undefined) return null;
  if (typeof v !== "object") return v;
  if (v._req) return { $r: v.id };
  if (v._sr) return { $sr: [v.req.id, v.num_computed, v.num_new] };
  if (v._t) return { $t: v._t };
  if (v._x) return { $x: v._x };
  if (depth > 2) return { $x: "…" };
  if (Array.isArray(v)) return v.slice(0, 24).map((x) => ser(x, depth + 1));
  const o = {};
  for (const [k, x] of Object.entries(v)) if (!k.startsWith("_")) o[k] = ser(x, depth + 1);
  return o;
}

const serFrame = (f) => {
  const vars = {};
  for (const [k, v] of Object.entries(f.L)) if (v !== undefined && !k.startsWith("_")) vars[k] = ser(v);
  return { fn: f.fn, label: f.label, node: f.node, vars, iters: { ...f.iters }, totals: { ...f.totals }, dec: { ...f.dec }, notes: { ...f.notes } };
};

// ---------------------------------------------------------------- threads

class Thread {
  constructor(name, kind) {
    this.name = name;
    this.kind = kind;
    this.stack = [];
    this.state = "ready";
    this.done = false;
    this.gen = null;
  }
}

export class Interp {
  constructor(G, cap = 9000) {
    this.G = G;
    this.cap = cap;
    this.events = [];
    this.threads = [];
    this.cur = null;
    this.pending = [];
    this.resume = [];
    this.lastWorld = null;
    this.lastWorldStr = "";
    this.halted = null;
    G.driver = this;
  }

  // ------------------------------------------------ threads API (used by closures)

  spawn(name, kind, fn, args, { now = false } = {}) {
    const t = new Thread(name, kind);
    t.gen = this.call(t, fn, args);
    this.threads.push(t);
    if (now) this.pending.push(t);
    return t;
  }
  wake(t, now = false) {
    if (!t || t.done) return;
    t.state = "ready";
    if (now) this.pending.push(t);
  }
  thread(name) {
    return this.threads.find((t) => t.name === name);
  }

  // ------------------------------------------------ interpreter

  *call(th, fn, args, label) {
    const spec = FNS[fn];
    if (!spec) throw new Error(`no flowchart for ${fn}`);
    const f = { fn, label, L: {}, node: null, iters: {}, totals: {}, dec: {}, notes: {} };
    spec.params.forEach((p, i) => (f.L[p] = args ? args[i] : undefined));
    th.stack.push(f);
    let ret;
    try {
      yield* this.block(th, spec.body, f);
    } catch (e) {
      if (e instanceof Ret) ret = e.v;
      else throw e;
    }
    th.stack.pop();
    return ret;
  }

  *block(th, nodes, f) {
    for (const n of nodes) yield* this.exec(th, n, f);
  }

  // forget the notes / decisions of a loop body before its next iteration
  clear(f, n) {
    for (const id of n.desc) {
      delete f.notes[id];
      delete f.dec[id];
      if (f.iters[id] !== undefined) {
        delete f.iters[id];
        delete f.totals[id];
      }
    }
  }

  *mark(th, f, n, note) {
    f.node = n.id;
    if (note !== undefined) f.notes[n.id] = note;
    this.record(th);
    yield STEP;
  }

  // a call inside a loop / if header: mark the header, run the callee under it
  *headerCall(th, f, n) {
    const args = n.call.args ? n.call.args(f.L, this.G) : [];
    yield* this.mark(th, f, n, `→ ${n.call.fn.split(".").pop()}(${args.map((a) => fmt(a)).join(", ")})`);
    return yield* this.call(th, n.call.fn, args, n.call.label);
  }

  *exec(th, n, f) {
    const G = this.G, L = f.L;
    switch (n.t) {
      case "S": {
        const note = n.run ? n.run(L, G) : undefined;
        yield* this.mark(th, f, n, note);
        return;
      }
      case "IF": {
        const r = n.call ? yield* this.headerCall(th, f, n) : undefined;
        const v = !!n.test(L, G, r);
        f.dec[n.id] = v;
        yield* this.mark(th, f, n, n.label ? n.label(L, G, v, r) : v ? "True" : "False");
        if (v) yield* this.block(th, n.then, f);
        else if (n.els) yield* this.block(th, n.els, f);
        return;
      }
      case "WHILE": {
        let k = 0;
        for (;;) {
          this.clear(f, n);
          const r = n.call ? yield* this.headerCall(th, f, n) : undefined;
          const v = !!n.test(L, G, r);
          if (v) k++;
          f.iters[n.id] = k;
          yield* this.mark(th, f, n, n.label ? n.label(L, G, v, k, r) : v ? `True · iter ${k}` : "False · exit");
          if (!v) return;
          try {
            yield* this.block(th, n.body, f);
          } catch (e) {
            if (e === BREAK) return;
            if (e !== CONTINUE) throw e;
          }
        }
      }
      case "FOR": {
        const ret = n.call ? yield* this.headerCall(th, f, n) : undefined;
        const items = [...n.iter(L, G, ret)];
        f.totals[n.id] = items.length;
        for (let k = 0; k < items.length; k++) {
          this.clear(f, n);
          n.bind(L, items[k], k, G);
          f.iters[n.id] = k + 1;
          yield* this.mark(th, f, n, n.label ? n.label(L, G, k) : `iter ${k + 1}/${items.length}`);
          try {
            yield* this.block(th, n.body, f);
          } catch (e) {
            if (e === BREAK) break;
            if (e !== CONTINUE) throw e;
          }
        }
        yield* this.mark(th, f, n, items.length ? `done · ${items.length} iter` : "empty → skip");
        return;
      }
      case "CALL": {
        if (n.when && !n.when(L, G)) {
          yield* this.mark(th, f, n, n.otherwise ? n.otherwise(L, G) : "skipped");
          return;
        }
        const args = n.args ? n.args(L, G) : [];
        yield* this.mark(th, f, n, `→ ${n.fn.split(".").pop()}(${args.map((a) => fmt(a)).join(", ")})`);
        const r = yield* this.call(th, n.fn, args, n.label);
        const note = n.assign ? n.assign(L, r, G) : undefined;
        yield* this.mark(th, f, n, note ?? `← ${fmt(r)}`);
        if (n.returns) throw new Ret(n.returns === true ? r : n.returns(L, r, G));
        return;
      }
      case "RET": {
        const v = n.value ? n.value(L, G) : undefined;
        yield* this.mark(th, f, n, n.value ? `→ ${fmt(v)}` : undefined);
        throw new Ret(v);
      }
      case "BRK":
        yield* this.mark(th, f, n);
        throw BREAK;
      case "CONT":
        yield* this.mark(th, f, n);
        throw CONTINUE;
      case "AWAIT": {
        let blocked = false;
        for (;;) {
          if (n.ready(L, G)) {
            const note = n.take(L, G);
            yield* this.mark(th, f, n, note);
            return;
          }
          if (n.empty && n.empty(L, G)) {
            yield* this.mark(th, f, n, "queue.Empty → break");
            throw BREAK;
          }
          if (!blocked) {
            blocked = true;
            f.node = n.id;
            f.notes[n.id] = n.waitNote || "⏸ blocked";
            th.state = "blocked";
            this.record(th);
          }
          th.state = "blocked";
          yield SUSPEND;
        }
      }
    }
  }

  // ------------------------------------------------ recording

  record(th) {
    const G = this.G;
    const w = snapshotWorld(G);
    const s = JSON.stringify(w);
    if (s !== this.lastWorldStr) {
      this.lastWorldStr = s;
      this.lastWorld = w;
    }
    const top = th.stack[th.stack.length - 1];
    this.events.push({
      thread: th.name,
      depth: th.stack.length,
      fn: top.fn,
      node: top.node,
      note: top.notes[top.node],
      group: G.stepNo,
      hl: G.hl || {},
      threads: this.threads.map((t) => ({ name: t.name, kind: t.kind, state: t === th ? "running" : t.done ? "done" : t.state, frames: t.stack.map(serFrame) })),
      world: this.lastWorld,
    });
    G.hl = null;
    if (this.events.length >= this.cap) throw new Error("event cap");
  }

  // ------------------------------------------------ driver

  pick() {
    while (this.pending.length) {
      const t = this.pending.shift();
      if (t.done) continue;
      if (this.cur && this.cur !== t && !this.cur.done && !this.resume.includes(this.cur)) this.resume.push(this.cur);
      this.cur = t;
      t.state = "ready";
      return t;
    }
    if (this.cur && !this.cur.done && this.cur.state === "ready") return this.cur;
    while (this.resume.length) {
      const t = this.resume.pop();
      if (!t.done && t.state === "ready") return (this.cur = t);
    }
    const ready = this.threads.find((t) => !t.done && t.state === "ready");
    if (ready) return (this.cur = ready);
    // everyone is blocked: let the next client arrive (time moves on)
    if (this.G.nextClient && this.G.nextClient()) return this.pick();
    return null;
  }

  run(entry) {
    const main = new Thread("main thread", "main");
    main.gen = this.call(main, entry, []);
    this.threads.push(main);
    this.cur = main;
    try {
      for (;;) {
        const t = this.pick();
        if (!t) break;
        const r = t.gen.next();
        if (r.done) {
          t.done = true;
          continue;
        }
        if (r.value === SUSPEND) t.state = "blocked";
        if (this.G.stepNo > this.G.cfg.maxSteps) {
          this.halted = `stopped after ${this.G.cfg.maxSteps} engine steps`;
          break;
        }
      }
    } catch (e) {
      this.halted = e.message === "event cap" ? `stopped at ${this.cap} events` : e.message;
      if (e.message !== "event cap") console.error(e);
    }
    return { events: this.events, hashInfo: this.G.hashInfo, halted: this.halted };
  }
}

// Executable flowcharts: every Python function in tinyserve.py is written once
// as a tree of nodes. Each node carries the Python text that is drawn in the
// diagram and a small JS closure that does what that line does, so the diagram
// and the trace can never disagree.
//
//   S(text, run)                      a statement            run(L, G) -> note?
//   IF(text, test, then, els, opt)    a decision             test(L, G, ret) -> bool
//   WHILE(text, test, body, opt)      a while loop
//   FOR(text, iter, bind, body, opt)  a for loop             iter(L, G, ret) -> items, bind(L, item, k, G)
//                                     opt.call = { fn, args } on IF / WHILE / FOR: the header itself
//                                     calls fn first (for o in self.step(), while not allocate(...))
//                                     opt.label(L, G, ...) -> the note drawn on the header
//   CALL(text, fn, args, assign, opt) a call that can be stepped into
//                                     opt: when / otherwise (conditional call), returns, label
//   RET(text, value)                  return
//   BRK(), CONT()                     break / continue
//   AWAIT(text, opt)                  a blocking get: opt.ready / take / empty (non-blocking -> break)
//   GRP(title, body, viz)             a block: consecutive lines that do one job. Drawn as one box
//                                     (title + a picture of what it does) until opened. blocks.js
//                                     groups each function's body, so program.js stays plain Python.
//
// L is the frame's locals (shown in the diagram), G is the whole simulated world.

export const FNS = {};

export const S = (text, run) => ({ t: "S", text, run });
export const IF = (text, test, then, els, opt = {}) => ({ t: "IF", text, test, then, els, ...opt });
export const WHILE = (text, test, body, opt = {}) => ({ t: "WHILE", text, test, body, ...opt });
export const FOR = (text, iter, bind, body, opt = {}) => ({ t: "FOR", text, iter, bind, body, ...opt });
export const CALL = (text, fn, args, assign, opt = {}) => ({ t: "CALL", text, fn, args, assign, ...opt });
export const RET = (text, value) => ({ t: "RET", text, value });
export const BRK = (text = "break") => ({ t: "BRK", text });
export const CONT = (text = "continue") => ({ t: "CONT", text });
export const AWAIT = (text, opt) => ({ t: "AWAIT", text, ...opt });
export const GRP = (title, body, viz) => ({ t: "GRP", title, body, viz });

export const NODE = {};

// mod picks the colour family: main, engine, sched, kv, runner, model, sample, detok, async
export function def(name, { sig, params = [], mod, body, doc }) {
  FNS[name] = { name, sig, params, mod, body, doc };
}

// wrap each function's body into blocks (spec from blocks.js), then number every node
export function finalize(BLOCKS) {
  for (const spec of Object.values(FNS)) {
    if (spec.done) continue;
    spec.done = true;
    const b = BLOCKS[spec.name];
    spec.body = b ? groupNodes(spec.body, b, spec.name) : [GRP(spec.doc || spec.name.split(".").pop(), spec.body)];
    number(spec.name, spec.body);
  }
}

// part = list of [count, title, viz] | { keep: 1, body|then: part } (a container whose inside is grouped)
function groupNodes(nodes, part, name) {
  const out = [];
  let k = 0;
  for (const p of part) {
    if (Array.isArray(p)) {
      const [n, title, viz] = p;
      out.push(GRP(title, nodes.slice(k, k + n), viz));
      k += n;
    } else {
      const n = nodes[k++];
      if (p.body) n.body = groupNodes(n.body, p.body, name);
      if (p.then) n.then = groupNodes(n.then, p.then, name);
      out.push(n);
    }
  }
  if (k !== nodes.length) throw new Error(`blocks for ${name} cover ${k} of ${nodes.length} lines`);
  return out;
}

function number(name, body) {
  let k = 0;
  const walk = (nodes) => {
    const ids = [];
    for (const n of nodes) {
      n.id = `${name}#${k++}`;
      NODE[n.id] = n;
      ids.push(n.id);
      const kids = [n.body, n.then, n.els].filter(Boolean).flatMap(walk);
      n.desc = kids;
      ids.push(...kids);
    }
    return ids;
  };
  walk(body);
}

// the callee each kind of node can expand into
export const calleeOf = (n) => (n.t === "CALL" ? n.fn : n.call ? n.call.fn : null);

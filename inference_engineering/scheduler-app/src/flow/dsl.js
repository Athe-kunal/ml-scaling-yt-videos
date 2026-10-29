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

// mod picks the colour family: main, engine, sched, kv, runner, model, sample, detok, async
export function def(name, { sig, params = [], mod, body, doc }) {
  let k = 0;
  const walk = (nodes) => {
    const ids = [];
    for (const n of nodes) {
      n.id = `${name}#${k++}`;
      ids.push(n.id);
      const kids = [n.body, n.then, n.els].filter(Boolean).flatMap(walk);
      n.desc = kids;
      ids.push(...kids);
    }
    return ids;
  };
  walk(body);
  FNS[name] = { name, sig, params, mod, body, doc };
}

// the callee each kind of node can expand into
export const calleeOf = (n) => (n.t === "CALL" ? n.fn : n.call ? n.call.fn : null);

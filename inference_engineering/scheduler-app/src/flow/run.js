import "./program.js";
import { samplingParams } from "./program.js";
import { Interp } from "./interp.js";
import { makeWorld } from "./world.js";

// Build the world, wire up client arrivals for serve mode, and trace main().
export function runTrace(cfg) {
  const G = makeWorld(cfg);
  const it = new Interp(G);

  // serve: a client's HTTP request reaches AsyncEngine.generate on the event loop
  const start = (c) => {
    c.started = true;
    it.spawn(`client ${c.id}`, "client", "AsyncEngine.generate", [[...c.prompt], samplingParams(c.id)], { now: true });
  };
  // clients due by this engine step arrive at the top of _loop
  G.arrivals = () => {
    for (const c of G.clients) if (!c.started && c.arrive <= G.stepNo) start(c);
  };
  // everyone blocked (engine idle on the inbox): the next client shows up early
  G.nextClient = () => {
    if (cfg.mode !== "serve" || !G.async) return false;
    const c = G.clients.find((x) => !x.started);
    if (!c) return false;
    start(c);
    return true;
  };

  return it.run("main");
}

// Bombadil spec for the ViT / CLIP stepper.
// Run (dev server must be up on :5183):  make vit-fuzz
import { extract, always, now, next } from "@antithesishq/bombadil";

// 1. Built-in checks: no uncaught exceptions, no unhandled rejections,
//    no console.error, no 4xx/5xx — plus default click/navigation actions.
export * from "@antithesishq/bombadil/browser/defaults";

// ── Extractors: pull plain JSON out of the DOM on every captured state ──

// "step 3 / 12 · ← →"  →  { n: 3, total: 12 }
const counter = extract((s) => {
  const el = [...s.document.querySelectorAll("span")].find((e) =>
    /^step \d+ \/ \d+/.test(e.textContent ?? ""),
  );
  const m = el?.textContent?.match(/^step (\d+) \/ (\d+)/);
  return m ? { n: Number(m[1]), total: Number(m[2]) } : null;
});

const btnDisabled = (label: string) =>
  extract((s) => {
    const b = [...s.document.querySelectorAll("button")].find((e) =>
      (e.textContent ?? "").includes(label),
    );
    return b ? (b as HTMLButtonElement).disabled : null;
  });
const prevDisabled = btnDisabled("prev");
const nextDisabled = btnDisabled("next");

const view = extract((s) => (s.window.location.hash === "#clip" ? "clip" : "vit"));

const katexErrors = extract((s) => s.document.querySelectorAll(".katex-error").length);

const overflowX = extract(
  (s) => s.document.documentElement.scrollWidth - s.window.innerWidth,
);

// ── Properties: things that must hold in EVERY state Bombadil reaches ──

// Step counter is always present and in range.
export const stepCounterInRange = always(() => {
  const c = counter.current;
  return c !== null && c.n >= 1 && c.n <= c.total;
});

// "◀ prev" is disabled exactly on the first step, "next ▶" exactly on the last.
export const navButtonsMatchStep = always(() => {
  const c = counter.current;
  if (!c) return true; // covered by stepCounterInRange
  return prevDisabled.current === (c.n === 1) && nextDisabled.current === (c.n === c.total);
});

// Every LaTeX string in the flowchart / shape card actually parses.
export const noKatexErrors = always(() => katexErrors.current === 0);

// Layout never spills sideways (a classic "UI not properly built" bug).
export const noHorizontalOverflow = always(() => overflowX.current <= 1);

// Switching ViT ↔ CLIP resets to step 1 (App remounts the view via key=).
export const viewSwitchResetsStep = always(
  now(() => {
    const before = view.current;
    return next(() => view.current === before || counter.current?.n === 1);
  }),
);

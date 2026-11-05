import { useEffect, useState } from "react";

// Tiny animation helpers for the "watch the variable change" scenes.
// useProgress(key) runs 0 → 1 every time `key` changes (a new step arrives), so a
// scene can draw the value BEFORE the line, morph it, and land on the value AFTER.

export const clamp01 = (x) => Math.max(0, Math.min(1, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
export const logit = (s) => Math.log(s / (1 - s));

export function useProgress(key, ms = 1400) {
  const [st, setSt] = useState({ key, p: 0 });
  useEffect(() => {
    let raf, t0;
    const tick = (now) => {
      if (t0 == null) t0 = now;
      const q = Math.min(1, (now - t0) / ms);
      setSt({ key, p: q });
      if (q < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [key, ms]);
  return st.key === key ? st.p : 0; // a new key starts from 0, never from the old end state
}

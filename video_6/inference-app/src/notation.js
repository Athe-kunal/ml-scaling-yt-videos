import { T } from "./theme";

export const AXIS_COLOR = {
  Y: T.accent, Z: T.accent2, YZ: T.wire,
  // 3-axis ICI mesh (x,y,z), lower-case to match the paper's notation, used
  // by the MLP weight-stationary view: single axes get their own hue, any
  // multi-axis combo (xy/xz/xyz) shares a neutral "mixed" colour since no
  // single hue can represent 3+ axes at once.
  x: T.bad, y: T.accent, z: T.accent2, xy: T.soft, xz: T.soft, yz: T.wire, xyz: T.soft,
};

// "Q[B_Z, N_Y]" -> html with <sub>, and **AllToAll** -> <b>.
export function codeHtml(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/_\{([^}]+)\}/g, "<sub>$1</sub>")
    .replace(/_([A-Za-z]+)/g, "<sub>$1</sub>");
}

// "B_Z,K_Y,M {U_YZ}" -> { dims: [{name, shard}], unreduced: "YZ" }
export function parseDims(d) {
  const [dims, unreduced] = d.split(" {");
  return {
    dims: dims.split(",").map((x) => {
      const [name, shard] = x.split("_");
      return { name, shard: shard || "" };
    }),
    unreduced: unreduced ? unreduced.replace("}", "").replace("U_", "") : "",
  };
}

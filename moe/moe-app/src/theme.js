export const T = {
  bg: "#0A0E14",
  panel: "#161D29",
  well: "#10151F",
  ink: "#F5F8FB",
  soft: "#C7D1DD",
  dim: "#94A1B2",
  rule: "#334053",
  accent: "#6EF3DD",
  accent2: "#FF8AC6",
  wire: "#FFC94D",
  bad: "#FF8A8A",
};

export const mono = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace";
export const sans = "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif";

export const btn = (on, disabled) => ({
  fontFamily: mono, fontSize: 12.5, height: 34, padding: "0 14px", borderRadius: 7, cursor: disabled ? "default" : "pointer",
  border: `1px solid ${on ? T.accent : T.rule}`, background: on ? `${T.accent}18` : T.well, color: disabled ? T.rule : on ? T.accent : T.ink,
});

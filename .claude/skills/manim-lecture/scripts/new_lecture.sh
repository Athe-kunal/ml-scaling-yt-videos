#!/bin/bash
# usage: new_lecture.sh <app-dir> <make-name> "<Lecture title>" [Makefile]
#   copies the scaffold into <app-dir>, creates its uv venv, appends the make block.
set -euo pipefail
SKILL="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$1"; NAME="$2"; TITLE="$3"; MK="${4:-Makefile}"
[ -e "$DIR" ] && { echo "refusing: $DIR exists"; exit 1; }
mkdir -p "$DIR"
cp -R "$SKILL/assets/scaffold/." "$DIR/"
sed -i '' "s/TOPIC-manim/$NAME/" "$DIR/pyproject.toml" 2>/dev/null || sed -i "s/TOPIC-manim/$NAME/" "$DIR/pyproject.toml"
VAR=$(echo "$NAME" | tr 'a-z-' 'A-Z_')
grep -q "^$NAME:" "$MK" 2>/dev/null && echo "make target $NAME already exists — not appending" || \
  sed -e "s|__NAME__|$NAME|g" -e "s|__VAR__|$VAR|g" -e "s|__DIR__|$DIR|g" -e "s|__TITLE__|$TITLE|g" \
      "$SKILL/assets/Makefile.snippet" >> "$MK"
# own venv, never a member of a root uv workspace
(cd "$DIR" && uv sync -q)
grep -q '^Q ?=' "$MK" || echo "NOTE: Makefile has no 'Q ?= --hd' default — add it near the top."
echo "scaffold ready in $DIR · make $NAME-test · make $NAME Q=-l"

#!/bin/bash
# usage: sheet.sh <app-dir> <scene-glob> <out.png> [cols]
#   last frame of every beat (= what is on screen when you stop pressing →), tiled in lecture order
cd "$1"; T=$(mktemp -d); i=0
for j in $(ls slides/$2*.json | sort); do
  for f in $(python3 -c "import json;[print(s['file']) for s in json.load(open('$j'))['slides']]"); do
    ffmpeg -loglevel error -y -sseof -0.05 -i "$f" -frames:v 1 -vf scale=640:-1 $T/$(printf %03d $i).png; i=$((i+1))
  done
done
cols=${4:-3}; n=$(ls $T | wc -l); rows=$(( (n+cols-1)/cols ))
ffmpeg -loglevel error -y -i $T/%03d.png -vf "tile=${cols}x${rows}:padding=6:color=white" -frames:v 1 "$3"
echo "$n beats → $3"

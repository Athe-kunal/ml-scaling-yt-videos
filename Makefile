.PHONY: video4 video4-install video5 video5-install

VIDEO4_DIR := video_4/transformer-flops-app
VIDEO5_DIR := video_5/parallelism-app
VIDEO4_PORT ?= 5174
VIDEO5_PORT ?= 5178

video4-install:
	cd $(VIDEO4_DIR) && npm install

video4:
	cd $(VIDEO4_DIR) && npm run dev -- --port $(VIDEO4_PORT)

video5-install:
	cd $(VIDEO5_DIR) && npm install

video5:
	cd $(VIDEO5_DIR) && VITE_TOPIC=$(NAME) npm run dev -- --port $(VIDEO5_PORT)

.PHONY: video6 video6-install
VIDEO6_DIR := video_6/inference-app
VIDEO6_PORT ?= 5179

video6-install:
	cd $(VIDEO6_DIR) && npm install

video6:
	cd $(VIDEO6_DIR) && npm run dev -- --port $(VIDEO6_PORT)

.PHONY: ring ring-install
RING_DIR := ring-attention/ring-app
RING_PORT ?= 5180

ring-install:
	cd $(RING_DIR) && npm install

ring:
	cd $(RING_DIR) && npm run dev -- --port $(RING_PORT)

.PHONY: inference inference-install
INFERENCE_DIR := inference_engineering/scheduler-app
INFERENCE_PORT ?= 5181

inference-install:
	cd $(INFERENCE_DIR) && npm install

inference:
	cd $(INFERENCE_DIR) && npm run dev -- --port $(INFERENCE_PORT)

.PHONY: moe moe-install
MOE_DIR := moe/moe-app
MOE_PORT ?= 5182

moe-install:
	cd $(MOE_DIR) && npm install

moe:
	cd $(MOE_DIR) && npm run dev -- --port $(MOE_PORT)

.PHONY: vit vit-install
VIT_DIR := vision-audio/vit-app
VIT_PORT ?= 5183

vit-install:
	cd $(VIT_DIR) && npm install

vit:
	cd $(VIT_DIR) && npm run dev -- --port $(VIT_PORT)

# Bombadil property-based UI fuzzing (start `make vit` in another terminal first)
vit-fuzz:
	cd $(VIT_DIR) && npx bombadil browser test http://localhost:$(VIT_PORT) bombadil/spec.ts \
		--time-limit=$(or $(T),3m) --output-path bombadil-out --output-path-overwrite

vit-inspect:
	cd $(VIT_DIR) && npx bombadil browser inspect bombadil-out

# PaperForge drills, served from vision-audio/paperforge-packs (packs live in this repo):
#   ViT / CLIP / SigLIP, LLaVA / DeepStack (NumPy), Whisper from scratch → Qwen2-Audio / Qwen3-Omni (PyTorch)
# vit-drills and whisper-drills open the same server, which lists all three packs.
.PHONY: vit-drills vit-drills-validate whisper-drills whisper-drills-validate
PAPERFORGE_DIR ?= $(HOME)/Desktop/DEV/Learning/paperforge
VIT_PACKS_DIR := $(CURDIR)/vision-audio/paperforge-packs

# own ports, so it never talks to another PaperForge runner (which would list every pack)
DRILLS_APP_PORT ?= 5175
DRILLS_RUNNER_PORT ?= 8766

vit-drills:
	PAPERFORGE_PACKS=$(VIT_PACKS_DIR) PAPERFORGE_APP_PORT=$(DRILLS_APP_PORT) PAPERFORGE_RUNNER_PORT=$(DRILLS_RUNNER_PORT) $(PAPERFORGE_DIR)/start.sh

vit-drills-validate:
	$(PAPERFORGE_DIR)/.venv/bin/python $(PAPERFORGE_DIR)/skill/paper-drills/scripts/validate_pack.py $(VIT_PACKS_DIR)/vit-clip-siglip-numpy
	$(PAPERFORGE_DIR)/.venv/bin/python $(PAPERFORGE_DIR)/skill/paper-drills/scripts/validate_pack.py $(VIT_PACKS_DIR)/llava-deepstack-numpy

whisper-drills: vit-drills

whisper-drills-validate:
	$(PAPERFORGE_DIR)/.venv/bin/python $(PAPERFORGE_DIR)/skill/paper-drills/scripts/validate_pack.py $(VIT_PACKS_DIR)/whisper

.PHONY: whisper whisper-install
WHISPER_DIR := vision-audio/whisper-app
WHISPER_PORT ?= 5184

whisper-install:
	cd $(WHISPER_DIR) && npm install

whisper:
	cd $(WHISPER_DIR) && npm run dev -- --port $(WHISPER_PORT)

# Manim lecture slides (ManimGL + manim-slides): → next, ← previous, Q quit, F fullscreen
# Each lesson is split into sections; only sections whose code changed re-render, in parallel.
#   make whisper-manim                   render what changed, then present all lessons
#   make whisper-manim L=L2              one lesson (any part of a scene name: L2_Fourier, probe, ...)
#   make whisper-manim Q=-l              fast 480p while editing (default: --hd 1080p)
#   make whisper-manim-html              same slides in the browser (fits any screen)
.PHONY: whisper-manim whisper-manim-html whisper-manim-install whisper-manim-render whisper-manim-test whisper-manim-clean
WM_DIR := vision-audio/whisper-manim
Q ?= --hd
WM_ARGS = $(L) --quality=$(Q)

whisper-manim-install:
	cd $(WM_DIR) && uv sync

whisper-manim-render:
	cd $(WM_DIR) && uv run python render.py $(WM_ARGS)

# desktop player: fullscreen + letterboxed on whatever display it opens on
whisper-manim:
	cd $(WM_DIR) && uv run python render.py $(WM_ARGS) --present

# browser version (reveal.js): scales to any window/projector, same ← / → keys
whisper-manim-html:
	cd $(WM_DIR) && uv run python render.py $(WM_ARGS) --html

whisper-manim-test:
	cd $(WM_DIR) && uv run pytest -q

whisper-manim-clean:
	rm -rf $(WM_DIR)/slides $(WM_DIR)/media $(WM_DIR)/videos

# ------------------------------------------------------------ Soft-MTP lecture (ManimGL + manim-slides, ← / →)
#   make mtp-manim                       render what changed, then present all 4 lessons
#   make mtp-manim L=L3                  one lesson (any part of a scene name: L3_Trick, backward, ...)
#   make mtp-manim Q=-l                  fast 480p while editing (default: --hd 1080p)
#   make mtp-manim-html                  same slides in the browser (T = contents sidebar)
.PHONY: mtp-manim mtp-manim-html mtp-manim-install mtp-manim-render mtp-manim-test mtp-manim-clean
MTP_DIR := mtp/soft-mtp-manim
MTP_ARGS = $(L) --quality=$(Q)

mtp-manim-install:
	cd $(MTP_DIR) && uv sync

mtp-manim-render:
	cd $(MTP_DIR) && uv run python render.py $(MTP_ARGS)

mtp-manim:
	cd $(MTP_DIR) && uv run python render.py $(MTP_ARGS) --present

mtp-manim-html:
	cd $(MTP_DIR) && uv run python render.py $(MTP_ARGS) --html

mtp-manim-test:
	cd $(MTP_DIR) && uv run pytest -q

mtp-manim-clean:
	rm -rf $(MTP_DIR)/slides $(MTP_DIR)/media $(MTP_DIR)/videos

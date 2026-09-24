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

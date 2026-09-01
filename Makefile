.PHONY: video4 video4-install video5 video5-install

VIDEO4_DIR := video_4/transformer-flops-app
VIDEO5_DIR := video_5/parallelism-app

video4-install:
	cd $(VIDEO4_DIR) && npm install

video4:
	cd $(VIDEO4_DIR) && npm run dev

video5-install:
	cd $(VIDEO5_DIR) && npm install

video5:
	cd $(VIDEO5_DIR) && VITE_TOPIC=$(NAME) npm run dev

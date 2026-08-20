.PHONY: video4 video4-install

VIDEO4_DIR := video_4/transformer-flops-app

video4-install:
	cd $(VIDEO4_DIR) && npm install

video4:
	cd $(VIDEO4_DIR) && npm run dev

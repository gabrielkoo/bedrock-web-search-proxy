build-BedrockWebSearchProxy:
	cp main.py run.sh "$(ARTIFACTS_DIR)/"
	chmod +x "$(ARTIFACTS_DIR)/run.sh"

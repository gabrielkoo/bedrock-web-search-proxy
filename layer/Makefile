ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
    PLATFORM := manylinux_2_28_aarch64
else
    PLATFORM := manylinux_2_28_x86_64
endif

build-DepsLayer:
	pip install \
		--platform $(PLATFORM) \
		--python-version 3.13 \
		--only-binary=:all: \
		--target "$(ARTIFACTS_DIR)/python" \
		-r requirements.txt

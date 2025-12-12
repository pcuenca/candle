PACKAGE := candle-sd15-bridge
CARGO   := cargo
TARGETS := aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim
LIBNAME := libcandle_sd15_bridge.a
OUTPUT  := CandleSD15.xcframework
HEADERS_SRC := candle-sd15-bridge/include
MODULEMAP := $(HEADERS_SRC)/module.modulemap
FEATURES ?= metal
FEATURES_FLAG := $(if $(FEATURES),--features $(FEATURES),)

.PHONY: all xcframework clean $(TARGETS:%=build-%)

all: xcframework

$(TARGETS:%=build-%): build-%:
	IPHONEOS_DEPLOYMENT_TARGET=26.0 $(CARGO) build -p $(PACKAGE) --release $(FEATURES_FLAG) --target $*

xcframework: $(TARGETS:%=build-%)
	rm -rf $(OUTPUT)
	xcodebuild -create-xcframework \
		-library target/aarch64-apple-ios/release/$(LIBNAME) \
		-headers $(HEADERS_SRC) \
		-library target/aarch64-apple-ios-sim/release/$(LIBNAME) \
		-headers $(HEADERS_SRC) \
		-library target/aarch64-apple-darwin/release/$(LIBNAME) \
		-headers $(HEADERS_SRC) \
		-output $(OUTPUT)

clean:
	rm -rf $(OUTPUT)

PACKAGE := candle-depth-bridge
CARGO   := cargo
TARGETS := aarch64-apple-darwin aarch64-apple-ios aarch64-apple-ios-sim
LIBNAME := libcandle_depth_bridge.a
OUTPUT  := CandleDepth.xcframework
HEADERS_SRC := candle-depth-bridge/include
MODULEMAP := $(HEADERS_SRC)/module.modulemap
FEATURES ?= metal
FEATURES_FLAG := $(if $(FEATURES),--features $(FEATURES),)

.PHONY: all xcframework clean $(TARGETS:%=build-%)

all: xcframework

$(TARGETS:%=build-%): build-%:
	$(CARGO) build -p $(PACKAGE) --release $(FEATURES_FLAG) --target $*

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

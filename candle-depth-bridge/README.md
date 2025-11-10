# candle-depth-bridge

This directory contains an adaptation of the Depth Anything V2 example to
provide a C interface that can be used from Swift.

## Building

Create an XCFramework (requires Xcode command-line tools):

```bash
make -f candle-depth-bridge/Makefile
```

## Model weights

`candle_depth_init` expects the assets directory to contain:

- `dinov2_vits14.safetensors`
- `depth_anything_v2_vits.safetensors`

If any of the files is not found, initialization returns a
`CandleDepthStatusCode`.

## C API

- `candle_depth_init` initializes the bridge. Set `use_metal` to enable the Metal
  backend (must compile with the `metal` feature). You probably want this.
- `candle_depth_infer` consumes a `CandleDepthRequest`. The image view must
  describe an RGB or RGBA buffer with `len = width * height * channels`. Set
  `use_color_map` to a non-zero value to receive a coloured depth map; otherwise
  a grayscale map is returned.
- `candle_depth_free_image` releases the heap allocation returned through
  `CandleDepthImage`.
- `candle_depth_last_error` surfaces the most recent error message.

The output image is always returned as RGBA8 data with a stride of 4 bytes and
matches the input width and height.

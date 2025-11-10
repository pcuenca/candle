#ifndef CANDLE_DEPTH_BRIDGE_H
#define CANDLE_DEPTH_BRIDGE_H

// Generated with cbindgen.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum CandleDepthStatusCode {
  Ok = 0,
  InvalidArgument = 1,
  NotInitialized = 2,
  AlreadyInitialized = 3,
  RuntimeError = 4,
  AssetRootMissing = 5,
  AssetDinov2Missing = 6,
  AssetDepthModelMissing = 7,
  LoadDinov2Failed = 8,
  LoadDepthModelFailed = 9,
  MetalUnavailable = 10,
} CandleDepthStatusCode;

typedef struct CandleDepthInitOptions {
  const char *asset_dir;
  uint8_t use_metal;
} CandleDepthInitOptions;

typedef struct CandleDepthImageView {
  const uint8_t *data;
  size_t len;
  uint32_t width;
  uint32_t height;
  uint32_t channels;
} CandleDepthImageView;

typedef struct CandleDepthRequest {
  struct CandleDepthImageView image;
  uint8_t use_color_map;
} CandleDepthRequest;

typedef struct CandleDepthImage {
  uint8_t *data;
  size_t len;
  size_t capacity;
  uint32_t width;
  uint32_t height;
  uint32_t channels;
} CandleDepthImage;

enum CandleDepthStatusCode candle_depth_init(const struct CandleDepthInitOptions *options);

bool candle_depth_is_ready(void);

enum CandleDepthStatusCode candle_depth_infer(const struct CandleDepthRequest *request,
                                              struct CandleDepthImage *out_image);

void candle_depth_free_image(struct CandleDepthImage *image);

const char *candle_depth_last_error(void);

#endif /* CANDLE_DEPTH_BRIDGE_H */

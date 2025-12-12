#ifndef CANDLE_SD15_BRIDGE_H
#define CANDLE_SD15_BRIDGE_H

// Generated with cbindgen.

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef enum CandleSdStatusCode {
  Ok = 0,
  InvalidArgument = 1,
  NotInitialized = 2,
  AlreadyInitialized = 3,
  RuntimeError = 4,
  AssetDirInvalid = 5,
  AssetRootMissing = 6,
  AssetTokenizerMissing = 7,
  AssetClipMissing = 8,
  AssetUnetMissing = 9,
  AssetVaeMissing = 10,
  LoadTokenizerFailed = 11,
  LoadClipFailed = 12,
  LoadUnetFailed = 13,
  LoadVaeFailed = 14,
  MetalUnavailable = 15,
} CandleSdStatusCode;

typedef struct CandleSdInitOptions {
  const char *asset_dir;
  uint8_t use_metal;
} CandleSdInitOptions;

typedef struct CandleSdRequest {
  const char *prompt;
  const char *negative_prompt;
  uint32_t steps;
  uint64_t seed;
  uint8_t use_seed;
} CandleSdRequest;

typedef struct CandleSdImage {
  uint8_t *data;
  size_t len;
  size_t capacity;
  uint32_t width;
  uint32_t height;
  uint32_t channels;
} CandleSdImage;

enum CandleSdStatusCode candle_sd_init(const struct CandleSdInitOptions *options);

bool candle_sd_is_ready(void);

enum CandleSdStatusCode candle_sd_generate(const struct CandleSdRequest *request,
                                           struct CandleSdImage *out_image);

void candle_sd_free_image(struct CandleSdImage *image);

const char *candle_sd_last_error(void);

#endif /* CANDLE_SD15_BRIDGE_H */

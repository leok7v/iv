#pragma once
#include "crt.h"

begin_c

typedef struct up_image_s {
    uint8_t* p; // pixels
    int32_t  w; // width
    int32_t  h; // height
    int32_t  s; // stride
    int32_t  c; // components (1,3,4)
} up_image_t;

typedef struct up_s {
    const up_image_t input;
    up_image_t output;
    up_image_t half; // downscaled image
} up_t;

typedef struct up_if {
    // upscale() both input and output images must be allocated by caller
    // seed must be odd and is needed for NN training, use 0x1 in debug
    void (*upscale)(up_t* u, uint64_t seed);
} up_if;

extern up_if up;

end_c

#pragma once
#include "crt.h"

begin_c

typedef struct up_s {
    const byte* input_pixels;
    int32_t iw; // width
    int32_t ih; // height
    int32_t is; // stride
    int32_t ic; // components
    byte* output_pixels;
    int32_t ow; // width
    int32_t oh; // height
    int32_t os; // stride
    int32_t oc; // components
} up_t;

typedef struct up_if {
    void (*upscale)(up_t* up, uint32_t* seed);
} up_if;

extern up_if up;

end_c

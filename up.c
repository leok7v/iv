/* Copyright (c) Dmitry "Leo" Kuznetsov 2021 see LICENSE for details */
#include "up.h"
#include "dense.h"

begin_c

static void upscale(up_t* u, uint32_t* unused(seed)) {
    // TODO: implement image upscaler
    int32_t w = u->input.w % 2 == 0 ? u->input.w : u->input.w - 1;
    int32_t h = u->input.h % 2 == 0 ? u->input.h : u->input.h - 1;
    up_image_t d = { // downscaled image
        .p = (uint8_t*)malloc(w / 2 * h / 2 * u->input.c),
        .w = w / 2,
        .h = h / 2,
        .s = w / 2 * u->input.c,
        .c = u->input.c
    };
    fatal_if_null(d.p); // no memory - almost impossible in UI apps
    traceln("TODO: downscale");
    traceln("TODO: train network");
    traceln("TODO: upscale thru inference");
    free(d.p);
}

extern up_if up = {
    .upscale = upscale
};

end_c

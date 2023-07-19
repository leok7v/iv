/* Copyright (c) Dmitry "Leo" Kuznetsov 2021 see LICENSE for details */
#include "up.h"

begin_c

static void upscale(up_t* img, uint32_t* seed) {
    // TODO: implement image upscaler
    (void)img; (void)seed;
}

extern up_if up = {
    .upscale = upscale
};

end_c

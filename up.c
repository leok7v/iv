/* Copyright (c) Dmitry "Leo" Kuznetsov 2021 see LICENSE for details */
#include "up.h"
#define fp_t float
#define fcl_type fp_t
#include "fcl.h"
#define _USE_MATH_DEFINES
#include <math.h>

begin_c

#include <stdint.h>
#include <math.h>

static fp_t max_h;
static fp_t max_s;
static fp_t max_i;

static void rgb_to_hsi(const uint8_t* rgb, fp_t* hsi) {
    // Normalize RGB values to the range [0, 1]
    fp_t r = rgb[0] / 255.0f;
    fp_t g = rgb[1] / 255.0f;
    fp_t b = rgb[2] / 255.0f;
    // Calculate the intensity (I)
    hsi[2] = (r + g + b) / 3.0f;
    // Calculate the saturation (S)
    fp_t min_rgb = (fp_t)fmin(r, fmin(g, b));
    fp_t sum_rgb = r + g + b;
    fp_t saturation = sum_rgb == 0 ? 0 : 1.0f - (3.0f * min_rgb / sum_rgb);
    hsi[1] = saturation;
    // Calculate the hue (H)
    if (saturation == 0.0) {
        hsi[0] = 0.0; // Hue is undefined for grayscale, so we set it to 0
    } else {
        fp_t numerator = 0.5f * ((r - g) + (r - b));
        fp_t denominator = (fp_t)sqrt((r - g) * (r - g) + (r - b) * (g - b));
        fp_t theta = (fp_t)acos(numerator / denominator);
        if (b <= g) {
            hsi[0] = theta;
        } else {
            hsi[0] = (fp_t)(2.0f * M_PI - theta);
        }
    }
    assert(!isnan(hsi[2]) && !isnan(hsi[1]) && !isnan(hsi[0]));
    assert(!isinf(hsi[2]) && !isinf(hsi[1]) && !isinf(hsi[0]));
    hsi[0] = (fp_t)(hsi[0] / (2 * M_PI)); // normalize to [0..1]
//  if (max_h < hsi[0]) { max_h = hsi[0]; traceln("max: %f %f %f", max_h, max_s, max_i); }
//  if (max_s < hsi[1]) { max_s = hsi[1]; traceln("max: %f %f %f", max_h, max_s, max_i); }
//  if (max_i < hsi[2]) { max_i = hsi[2]; traceln("max: %f %f %f", max_h, max_s, max_i); }
}

static void hsi_to_rgb(const fp_t* hsi, uint8_t* rgb) {
    fp_t hue = (fp_t)(hsi[0] * (2 * M_PI)); // denormalize [0..1] to [0..2 * M_PI]
    assert(0 <= hue && hue <= 2 * M_PI);
    fp_t hue_deg = (fp_t)(hue * 180 / M_PI); // Convert hue to degrees
    // Normalize Saturation and Intensity to the range [0, 1]
    fp_t saturation = (fp_t)fmax(0, fmin(1, hsi[1]));
    fp_t intensity = (fp_t)fmax(0, fmin(1, hsi[2]));
    // Convert hue to the range [0, 360)
    if (hue_deg >= 360.0f) {
        hue_deg -= 360.0f;
    }
    fp_t r, g, b;
    if (hue_deg >= 0.0 && hue_deg < 120.0) {
        b = (fp_t)(intensity * (1 - saturation));
        r = (fp_t)(intensity * (1 + (saturation * cos(hue)) / cos(M_PI / 3 - hue)));
        g = (fp_t)(3 * intensity - (r + b));
    } else if (hue_deg >= 120.0 && hue_deg < 240.0) {
        hue -= (fp_t)(2 * M_PI / 3.0);
        r = (fp_t)(intensity * (1 - saturation));
        g = (fp_t)(intensity * (1 + (saturation * cos(hue)) / cos(M_PI / 3 - hue)));
        b = (fp_t)(3 * intensity - (r + g));
    } else {
        hue -= (fp_t)(4 * M_PI / 3);
        g = (fp_t)(intensity * (1 - saturation));
        b = (fp_t)(intensity * (1 + (saturation * cos(hue)) / cos(M_PI / 3 - hue)));
        r = (fp_t)(3 * intensity - (g + b));
    }
    // Scale and convert to uint8_t
    rgb[0] = (uint8_t)(r * 255);
    rgb[1] = (uint8_t)(g * 255);
    rgb[2] = (uint8_t)(b * 255);
}

static void ground_truth(const up_t* u, fp_t* input, fp_t* output, int32_t n) {
    fp_t* i = input;
    fp_t* o = output;
    for (int32_t y = 0; y < u->half.h - 2; y++) {
        const int32_t y2 = y * 2;
        for (int32_t x = 0; x < u->half.w - 2; x++) {
            for (int32_t r = 0; r < 3; r++) { // row
                for (int32_t c = 0; c < 3; c++) { // column
                    rgb_to_hsi(u->half.p + (y + r) * u->half.s + (x + c) * u->half.c, i);
                    i += 3;
                }
            }
            const int32_t x2 = x * 2;
            for (int32_t r = 0; r < 2; r++) { // row
                for (int32_t c = 0; c < 2; c++) { // column
                    rgb_to_hsi(u->input.p + (y2 + r) * u->input.s + (x2 + c) * u->input.c, o);
                    o += 3;
                }
            }
        }
    }
    fatal_if(i != input  + n * 9 * 3);
    fatal_if(o != output + n * 4 * 3);
}

static void downscale(const up_image_t* s, up_image_t* d) {
    for (int32_t y = 0; y < d->h; y++) {
        uint8_t* dr = d->p + y * d->s; // destination row
        const int32_t  y2 = y * 2;
        const uint8_t* p0 = s->p + y2 * s->s;
        const uint8_t* p1 = p0 + s->s;
        for (int32_t x = 0; x < d->w; x++) {
            uint8_t* dp = dr + x * d->c; // destination pixels
            for (int32_t c = 0; c < d->c; c++) {
                const int32_t x2 = x * 2 * s->c + c;
                dp[c] = (uint8_t)(((uint32_t)
                    p0[x2] + p0[x2 + 1] + p1[x2] + p1[x2 + 1] + 3
                    ) / 4);
            }
        }
    }
}

static inline void int_swap(int32_t* a, int32_t* b) {
    int32_t t = *a; *a = *b; *b = t;
}

static void permute(int32_t* permutation, int32_t n, uint64_t* seed) {
    const int32_t m = n / 8;
    for (int i = 0; i < m; i++) {
        int32_t ix0 = (int32_t)(crt.random64(seed) % n);
        int32_t ix1 = (int32_t)(crt.random64(seed) % n);
        if (ix0 != ix1) { int_swap(permutation + ix0, permutation + ix1); }
    }
}

static void init_activation(fcl_t* nn) {
#if 1
    nn->activation_hidden = fcl.sigmoid;
    nn->activation_output = fcl.sigmoid;
#elif 0
    nn->activation_hidden = fcl.tanh;
    nn->activation_output = fcl.tanh;
#elif 0
    nn->activation_hidden = fcl.linear;
    nn->activation_output = fcl.linear;
#elif 0
    nn->activation_hidden = fcl.sigmoid;
    nn->activation_output = fcl.sigmoid;
#elif 0
    nn->activation_hidden = fcl.relu;
    nn->activation_output = fcl.sigmoid;
#elif 0
    nn->activation_hidden = fcl.relu;
    nn->activation_output = fcl.relu;
#elif 0
    nn->activation_hidden = fcl.leaky_relu;
    nn->activation_output = fcl.leaky_relu;
#elif 0
    nn->activation_hidden = fcl.elu;
    nn->activation_output = fcl.elu;
#elif 0
    nn->activation_hidden = fcl.swish;
    nn->activation_output = fcl.swish;
#else
    fatal_if("must choose one of activation functions");
#endif
}

static void upscale(up_t* u, uint64_t seed) {
    assert(u->input.w % 2 == 0, "expected even w: %d", u->input.w);
    assert(u->input.h % 2 == 0, "expected even h: %d", u->input.h);
    assert(u->input.c == 3, "dense layer is fixed constant size");
    downscale(&u->input, &u->half);
    traceln("collect ground truth");
    // [3x3] kernel at each pixel [1..h-1][1..w-1] x3 components
    const int32_t n = (u->half.h - 2) * (u->half.w - 2);
    int32_t* permutation = malloc(n * sizeof(int32_t));
    fatal_if_null(permutation);
    for (int32_t i = 0; i < n; i++) { permutation[i] = i; }
    fp_t* input  = malloc((3 * 3) * u->half.c  * n * sizeof(fp_t));
    fp_t* output = malloc((2 * 2) * u->input.c * n * sizeof(fp_t));
    fatal_if_null(input);
    fatal_if_null(output);
    ground_truth(u, input, output, n);
    traceln("train network");
    fp_t learning_rate = 0.5f;
    enum { inputs = 3 * 3 * 3, layers = 3, hidden = 8, outputs = 1, epochs = 10 };
    int64_t bytes = fcl_training_memory_size(inputs, layers, hidden, outputs);
    uint8_t* nn12 = malloc(bytes * 12);
    fatal_if_null(nn12);
    for (int32_t k = 0; k < 12; k++) {
        fcl_t* nn = (fcl_t*)(nn12 + bytes * k);
        fcl.init(nn, bytes, seed, inputs, layers, hidden, outputs);
        init_activation(nn);
        const fp_t weight_range = (fp_t)sqrt(6.0 / fcl_weights_count(inputs, layers, hidden, outputs));
        fcl.randomize(nn, weight_range);
        for (int32_t epoch = 0; epoch < epochs; epoch++) {
            fp_t max_loss = 0;
            fp_t loss_sum = 0;
            permute(permutation, n, &nn->seed);
            for (int32_t i = 0; i < n; i++) {
                const int32_t ix = permutation[i];
                fp_t loss = fcl.train(nn,
                    input + ix * 3 * 3 * 3,
                    output + ix * 2 * 2 * 3 + k, learning_rate);
                loss_sum += loss;
                if (loss > max_loss) { max_loss = loss; /* traceln("max_loss: %f", max_loss); */ }
//              traceln("[%d/%d] loss: %f", i, n, loss);
            }
            if (epoch == epochs - 1) {
                fp_t avg_loss = loss_sum / n;
                traceln("[%02d] loss max: %f avg: %f", k, max_loss, avg_loss);
            }
            learning_rate = learning_rate - learning_rate / 128;
        }
    }
    fp_t hsi[3 * 3 * 3] = { 0 }; // 27
    fp_t res[3] = { 0 }; // 12
    uint8_t rgb[3] = { 0 };
    for (int32_t y = 0; y < u->input.h - 2; y++) {
        for (int32_t x = 0; x < u->input.w - 2; x++) {
            for (int32_t r = 0; r < 3; r++) { // row
                for (int32_t c = 0; c < 3; c++) { // column
                    const int32_t yo = (y + r) * u->input.s;
                    const int32_t xo = (x + c) * u->input.c;
                    const int32_t hx = r * 3 * 3 + c * 3;
                    rgb_to_hsi(u->input.p + yo + xo, hsi + hx);
//                  traceln("[%03d:%d][%03d:%d] @%d %f, %f %f", y, r, x, c, hx, hsi[hx + 0], hsi[hx + 1], hsi[hx + 2]);
                }
            }
            for (int32_t r = 0; r < 2; r++) { // row
                for (int32_t c = 0; c < 2; c++) { // column
                    for (int32_t k = 0; k < 3; k++) { // hsi index
                        int32_t nx = r * 2 * 3 + c * 3 + k; // network index
//                      traceln("[%d][%d][%d] nn[%d]", r, c, k, nx);
                        fcl_t* nn = (fcl_t*)(nn12 + bytes * nx);
                        res[k] = fcl.inference(nn, hsi)[0];
                    }
                    hsi_to_rgb(res, rgb);
//                  traceln("%f, %f %f -> %02X%02X%02X", res[0], res[1], res[2], rgb[0], rgb[1], rgb[2]);
                    for (int32_t k = 0; k < 3; k++) { // rgb index
                        const int32_t y2 = y * 2 + r;
                        const int32_t x2 = x * 2 + c;
                        const int32_t ox = y2 * u->output.s + x2 * u->output.c + k;
                        u->output.p[ox] = rgb[k];
// const int32_t yo = (y + r) * u->input.s;
// const int32_t xo = (x + c) * u->input.c;
// u->output.p[ox] = u->input.p[yo + xo + (2 - k)];
                    }
                }
            }
        }
    }
    free(nn12);
    free(input);
    free(output);
    free(permutation);
}

extern up_if up = {
    .upscale = upscale
};

end_c

/* Copyright (c) Dmitry "Leo" Kuznetsov 2021 see LICENSE for details */
#include "up.h"
#include "dense.h"
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
    for (int32_t y = 1; y < u->half.h - 1; y++) {
        const int32_t y2 = (y - 1) * 2;
        for (int32_t x = 1; x < u->half.w - 1; x++) {
            for (int32_t r = 0; r < 3; r++) { // row
                for (int32_t c = 0; c < 3; c++) { // column
                    rgb_to_hsi(u->half.p + (y + r) * u->half.s + (x + c) * u->half.c, i);
                    i += 3;
                }
            }
            const int32_t x2 = (x - 1) * 2;
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

static void permute(int32_t* permutation, int32_t n, uint32_t* seed) {
    const int32_t m = n / 8;
    for (int i = 0; i < m; i++) {
        int32_t ix0 = crt.random32(seed) % n;
        int32_t ix1 = crt.random32(seed) % n;
        if (ix0 != ix1) { int_swap(permutation + ix0, permutation + ix1); }
    }
}

static void init_activation(dense_t* fcl) {
#if 0 // inf
    fcl->input_activation  = dense_linear_activation;
    fcl->input_derivative  = dense_linear_derivative;
    fcl->output_activation = dense_linear_activation;
    fcl->output_derivative = dense_linear_derivative;
#elif 0 // 10-12
    fcl->input_activation  = dense_tanh_activation;
    fcl->input_derivative  = dense_tanh_derivative;
    fcl->output_activation = dense_tanh_activation;
    fcl->output_derivative = dense_tanh_derivative;
#elif 0 // 1.5-1.6 ***
    fcl->input_activation  = dense_sigmoid_activation;
    fcl->input_derivative  = dense_sigmoid_derivative;
    fcl->output_activation = dense_sigmoid_activation;
    fcl->output_derivative = dense_sigmoid_derivative;
#elif 1
    fcl->input_activation  = dense_swish_activation;
    fcl->input_derivative  = dense_swish_derivative;
    fcl->output_activation = dense_sigmoid_activation;
    fcl->output_derivative = dense_sigmoid_derivative;
#elif 0 // 3.2
    fcl->input_activation  = dense_relu_activation;
    fcl->input_derivative  = dense_relu_derivative;
    fcl->output_activation = dense_relu_activation;
    fcl->output_derivative = dense_relu_derivative;
#elif 0 // 7 - 14
    fcl->input_activation  = dense_leaky_relu_activation;
    fcl->input_derivative  = dense_leaky_relu_derivative;
    fcl->output_activation = dense_leaky_relu_activation;
    fcl->output_derivative = dense_leaky_relu_derivative;
#elif 0 // HUGE
    fcl->input_activation  = dense_elu_activation;
    fcl->input_derivative  = dense_elu_derivative;
    fcl->output_activation = dense_elu_activation;
    fcl->output_derivative = dense_elu_derivative;
#elif 0 // HUGE
    fcl->input_activation  = dense_swish_activation;
    fcl->input_derivative  = dense_swish_derivative;
    fcl->output_activation = dense_swish_activation;
    fcl->output_derivative = dense_swish_derivative;
#else
    fatal_if("must choose one of activation functions");
#endif
}

static void upscale(up_t* u, uint32_t* seed) {
    assert(u->input.w % 2 == 0, "expected even w: %d", u->input.w);
    assert(u->input.h % 2 == 0, "expected even h: %d", u->input.h);
    assert(u->input.c == 3, "dense layer is fixed constant size");
    downscale(&u->input, &u->half);
    traceln("collect ground truth");
    // [3x3] kernel at each pixel [1..h-1][1..w-1] x3 components
    const int32_t n = (u->half.h - 2) * (u->half.w - 2);
    int32_t* permutation = malloc(n * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) { permutation[i] = i; }
    fp_t* input = malloc((9 * u->half.c) * n * sizeof(fp_t));
    fp_t* output = malloc((2 * 2) * u->input.c * n * sizeof(fp_t));
    fatal_if_null(input);
    fatal_if_null(output);
    ground_truth(u, input, output, n);
    traceln("train network");
    static dense_t fcl; // fully connected dense layer network
    fp_t learning_rate = 0.1f;
    dense.init(&fcl, seed);
    init_activation(&fcl);
    for (int32_t epoch = 0; epoch < 4; epoch++) {
        fp_t max_loss = 0;
        fp_t loss_sum = 0;
        permute(permutation, n, seed);
        for (int32_t i = 0; i < n; i++) {
            for (int32_t j = 0; j < 3 * 3 * 3; j++) {
                assert(!isnan(input[i * 3 * 3 * 3 + j]));
                assert(!isinf(input[i * 3 * 3 * 3 + j]));
            }
            for (int32_t j = 0; j < 2 * 2 * 3; j++) {
                assert(!isnan(output[i * 2 * 2 * 3 + j]));
                assert(!isinf(output[i * 2 * 2 * 3 + j]));
            }
            const int32_t ix = permutation[i];
            fp_t loss = dense.backward(&fcl,
                input + ix * 3 * 3 * 3,
                output + ix * 2 * 2 * 3, learning_rate);
            loss_sum += loss;
            if (loss > max_loss) { max_loss = loss; }
            // traceln("loss: %f", loss);
        }
        fp_t avg_loss = loss_sum / n;
        traceln("[%d] loss max: %f sqrt(max): %f avg: %f sqrt(avg): %f",
            epoch, max_loss, sqrt(max_loss), avg_loss, sqrt(avg_loss));
    }
    traceln("TODO: upscale thru inference");
    free(input);
    free(output);
}

extern up_if up = {
    .upscale = upscale
};

end_c

#pragma once
#include "crt.h"
#include <math.h>

begin_c

typedef float fp32_t;

#define fp_t fp32_t

enum { dense_inputs = 3 * 3 * 3, dense_outputs = 2 * 2 * 3, dense_neurons = 512 };

inline fp_t dense_sigmoid_activation(fp_t x) { return (fp_t)(1 / (1 + exp(-x))); }
inline fp_t dense_sigmoid_derivative(fp_t x) { fp_t sx = dense_sigmoid_activation(x); return sx * (1 - sx); }

inline fp_t dense_linear_activation(fp_t x) { return x; }
inline fp_t dense_linear_derivative(fp_t x) { (void)x; return 1; }

inline fp_t dense_tanh_activation(fp_t x) { return tanhf(x); }
inline fp_t dense_tanh_derivative(fp_t x) { return 1 - tanhf(x) * tanhf(x); }

inline fp_t dense_relu_activation(fp_t x) { return x > 0 ? x : 0.0f; }
inline fp_t dense_relu_derivative(fp_t x) { return x > 0 ? 1.0f : 0.0f; }

inline fp_t dense_leaky_relu_activation(fp_t x) { return x > 0 ? x : 0.01f * x; }
inline fp_t dense_leaky_relu_derivative(fp_t x) { return x > 0 ? 1 : 0.01f; }

inline fp_t dense_elu_activation(fp_t x) { return x >= 0 ? x : (fp_t)(0.01 * (exp(x) - 1)); }
inline fp_t dense_elu_derivative(fp_t x) { return x >= 0 ? 1 : (fp_t)(0.01 * exp(x)); }

inline fp_t dense_swish_activation(fp_t x) { return x * dense_sigmoid_activation(x); }
inline fp_t dense_swish_derivative(fp_t x) { fp_t sx = dense_sigmoid_activation(x); return sx + x * sx * (1 - sx); }

typedef struct dense_s {
    fp_t iw[dense_neurons][dense_inputs]; // input weights
    fp_t ib[dense_neurons];               // input bias
    fp_t (*input_activation)(fp_t x);
    fp_t (*input_derivative)(fp_t x);
    fp_t ow[dense_outputs][dense_neurons]; // output weights
    fp_t ob[dense_outputs];                // output bias
    fp_t(*output_activation)(fp_t x);
    fp_t(*output_derivative)(fp_t x);
} dense_t;

typedef struct dense_if {
    void (*init)(dense_t* dense, uint32_t* seed);
    void (*forward)(const dense_t* dense, const fp_t input[dense_inputs], fp_t output[dense_outputs]);
    fp_t (*backward)(dense_t* dense, const fp_t input[dense_inputs], const fp_t truth[dense_outputs], fp_t learning_rate);
} dense_if;

extern dense_if dense;

end_c

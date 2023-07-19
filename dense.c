#include "dense.h"

begin_c

static fp_t dot_product(const fp_t* a, const fp_t* b, size_t n) {
    fp_t dp = 0;
    for (size_t i = 0; i < n; i++) { dp += a[i] * b[i]; }
    return dp;
}

static void dense_forward(const dense_t* d, const fp_t input[dense_inputs], fp_t output[dense_outputs]) {
    // naive inference
    fp_t activations[dense_neurons] = { 0 };
    for (int i = 0; i < countof(d->iw); i++) {
        fp_t hidden_i = dot_product(d->iw[i], input, countof(d->iw[i]));
        activations[i] = d->input_activation(hidden_i + d->ib[i]);
    }
    for (int i = 0; i < countof(d->ow); i++) {
        output[i] = dot_product(d->ow[i], activations, countof(d->ow[i]));
        output[i] = d->output_activation(output[i] + d->ob[i]);
    }
}

static fp_t dense_backward(dense_t* d, const fp_t input[dense_inputs], const fp_t truth[dense_outputs]) {
    // naive backpropagation
    fp_t activations[dense_neurons] = { 0 };
    fp_t output[dense_outputs] = { 0 };
    fp_t output_error[dense_outputs] = { 0 };
    fp_t hidden_error[dense_neurons] = { 0 };
    fp_t loss = 0.0;
    dense_forward(d, input, output);
    // Calculate output layer error and loss
    for (int i = 0; i < dense_outputs; i++) {
        output_error[i] = (output[i] - truth[i]) * d->output_derivative(output[i]);
        loss += 0.5f * (output[i] - truth[i]) * (output[i] - truth[i]);
    }
    // Backward pass - calculate hidden layer error
    for (int i = 0; i < dense_neurons; i++) {
        fp_t error = 0;
        for (int j = 0; j < dense_outputs; j++) {
            error += d->ow[j][i] * output_error[j];
        }
        hidden_error[i] = error * d->input_derivative(activations[i]);
    }
    // Update output weights and biases
    for (int i = 0; i < dense_outputs; i++) {
        for (int j = 0; j < dense_neurons; j++) {
            d->ow[i][j] -= output_error[i] * activations[j];
        }
        d->ob[i] -= output_error[i];
    }
    // Update input weights and biases
    for (int i = 0; i < dense_neurons; i++) {
        for (int j = 0; j < dense_inputs; j++) {
            d->iw[i][j] -= hidden_error[i] * input[j];
        }
        d->ib[i] -= hidden_error[i];
    }
    return loss;
}

static fp_t dense_random(uint32_t* seed) {
    double r = crt.random32(seed) / (double)UINT32_MAX;
    return (fp_t)(r * 2 - 1.0);
}

static void dense_init(dense_t* d, uint32_t* seed) {
    for (int i = 0; i < countof(d->iw); i++) {
        for (int j = 0; j < countof(d->iw[i]); j++) {
            d->iw[i][j] = dense_random(seed);
        }
    }
    for (int i = 0; i < countof(d->ib); i++) {
        d->ib[i] = dense_random(seed);
    }
    for (int i = 0; i < countof(d->ow); i++) {
        for (int j = 0; j < countof(d->ow[i]); j++) {
            d->ow[i][j] = dense_random(seed);
        }
    }
    for (int i = 0; i < countof(d->ob); i++) {
        d->ob[i] = dense_random(seed);
    }
    d->input_activation = null;
    d->input_derivative = null;
    d->output_activation = null;
    d->output_derivative = null;
}

dense_if dense = {
    .init = dense_init,
    .forward = dense_forward,
    .backward = dense_backward
};

end_c

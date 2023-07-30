/* Copyright (c) 2023 Leo Kuznetsov
 * derived from
 * GENANN - Minimal C Artificial Neural Network
 * Copyright (c) 2015-2018 Lewis Van Winkle
 * http://CodePlea.com
 */

/*
     Usage:
     enum { inputs = 2, layers = 1, hidden = 2, outputs = 1};
     int64_t bytes = fcl_training_memory_size(inputs, layers, hidden, outputs);
     fcl_t* nn = (fcl_t*)malloc(bytes);
     if (nn != null) {
         uint64_t seed = 1;
         fcl.init(nn, bytes, seed, inputs, layers, hidden, outputs);
         nn->activation_hidden = fcl.sigmoid;
         fp_t inputs[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
         fp_t xor[4] = { 0, 1, 1, 0 };
         fp_t learning_rate = 3; // no idea why 3?
         for (int32_t epoch = 0; epoch < 500; epoch++) {
             for (int32_t i = 0; i < countof(inputs); i++) {
                 fcl.train(nn, inputs[i], &xor[i], learning_rate);
             }
         }
         for (int32_t i = 0; i < countof(inputs); i++) {
             fp_t* output = nn->inference(nn, input);
             printf("%.1f ^ %.1f = %.1f\n", inputs[i][0], inputs[i][1], *output);
         }
         free(nn);
     }
*/

#ifndef FCL_H
#define FCL_H
#include "crt.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef fp_t
#define fp_t float
#endif

typedef fp_t (*fcl_activation_t)(fp_t a);

// inference()
//
// state for first hidden layer is the output of input layer as:
//    state[0][i] = activation_hidden(sum(input[j] * iw[j + 1]) + ihw[0])
// where iw[0] is input bias
//
// for subsequent layers "k" > 0
//    state[next][i] = activation_hidden(sum(state[k - 1][j] * hw[k - 1][i][j + 1]) + hw[k - 1][i][0])
// hw[k - 1][i][0] is hidden layer neuron bias
//
// for inference next = (k + 1) % 2
// for training  next = k + 1
//
// output[i] = activation_output(sum(state[layers][j] * hw[layers][i][j + 1]) + hw[layers][i][0])
// note dimension of hw[layers + 1]

typedef /* begin_packed */ struct fcl_s {
    int64_t inputs;  // number elements in input vector
    int64_t layers;  // number of hidden layers, must be >= 1
    int64_t hidden;  // number of neurons in each hidden layer
    int64_t outputs; // number elements in output vector
    // iw and hw are located contiguously in memory and can be written out
    // as chunk of binary data
    // for each hidden neuron of first layer bias and weight for each input element
    fp_t* iw; // [hidden][inputs + 1] == null if layers == 0 && hidden == 0
    // for each neuron of each hidden layer bias and weights vector of previous layer weights:
    fp_t* hw;  // [layers][hidden][hidden + 1] or null for layers == 0
    fp_t* ow;  // [outputs][hidden + 1] output weights
    fp_t* output; // [outputs] elements of output vector
    // training only data:
    fp_t* state; // [2][hidden] for inference() | [layers][hidden] for train()
    // delta == null for inference only fcl
    fp_t* delta; // null inference() | [layers][hidden] for train()
    // output deltas:
    fp_t* od;    // null inference() | [outputs] for train()
    // initial seed for random64() generator used for this network
    uint64_t seed;
    // for known fcl.sigmoid, tanh, relu, linear activation functions
    // nn.train() knows the derivatives
    fcl_activation_t activation_hidden; // activation function to use for hidden neurons
    fcl_activation_t derivative_hidden; // derivative of hidden activation function
    fcl_activation_t activation_output; // activation function to use for output elements
    fcl_activation_t derivative_output; // derivative of output activation function
} /* end_packed */ fcl_t;

typedef struct fcl_if { // interface
    void (*init)(fcl_t* nn, int64_t bytes, uint64_t seed,
                 int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs);
    fp_t (*random)(fcl_t* nn, fp_t weight_range); // [0..1] random based on nn->seed
    // randomize initializes each wieight and bias [-weight_range/2..+weight_range/2]
    void (*randomize)(fcl_t* nn, fp_t weight_range);
    // inference() returns pointer to output array of elements
    fp_t* (*inference)(fcl_t* nn, const fp_t* input);
    fp_t  (*train)(fcl_t* nn, const fp_t* input, const fp_t* truth,
                  fp_t learning_rate); // returns loss
    // available activation functions:
    const fcl_activation_t sigmoid;
    const fcl_activation_t tanh;
    const fcl_activation_t relu;
    const fcl_activation_t linear;
    const fcl_activation_t leaky_relu;
    const fcl_activation_t elu;
    const fcl_activation_t swish;
} fcl_if;

extern fcl_if fcl;

// Memory management helpers.
// All counts are a number of fp_t elements not bytes

#define fcl_iw_count(inputs, hidden) ((hidden) == 0 ? \
    0 : ((hidden) * ((inputs) + 1)))

#define fcl_hw_count(layers, hidden) ((layers) == 0 ? \
    0 : (((layers) - 1)  * (hidden) * ((hidden) + 1)))

#define fcl_ow_count(inputs, hidden, outputs) ((hidden) == 0 ? \
    (outputs) * ((inputs) + 1) : ((outputs) * ((hidden) + 1)))

#define fcl_weights_count(inputs, layers, hidden, outputs) (            \
   (layers > 0 ?                                                        \
   (fcl_iw_count(inputs, hidden) + fcl_hw_count(layers, hidden)) : 0) + \
    fcl_ow_count(inputs, hidden, outputs)                  )

#define fcl_inference_state_count(hidden) (2 * (hidden))
#define fcl_training_state_count(layers, hidden) ((layers) * (hidden))
#define fcl_training_delta_count(layers, hidden) ((layers) * (hidden))

// input, [layers][hidden][hidden+1], output
#define fcl_weights_memory_iho(inputs, layers, hidden, outputs) \
    fp_t iw[hidden][inputs + 1];                                \
    fp_t hw[layers - 1][hidden][hidden + 1];                    \
    fp_t ow[outputs][hidden + 1]

// input, output
#define fcl_weights_memory_io(inputs, outputs) \
    fp_t ow[outputs][inputs + 1]

#define fcl_weight_memory_size(inputs, layers, hidden, outputs) \
    ((fcl_iw_count(inputs, hidden) +                            \
      fcl_hw_count(layers, hidden) +                            \
      fcl_ow_count(inputs, hidden, outputs)) * sizeof(fp_t))

#define fcl_base_memory_iho(inputs, layers, hidden, outputs) \
    fcl_weights_memory_iho(inputs, layers, hidden, outputs); \
    fp_t output[outputs]

#define fcl_base_memory_io(inputs, outputs)  \
    fcl_weights_memory_io(inputs,  outputs); \
    fp_t output[outputs]

#define fcl_base_memory_size(inputs, layers, hidden, outputs) ( \
    fcl_weight_memory_size(inputs, layers, hidden, outputs) +   \
    /* output: */(outputs) * sizeof(fp_t)                       \
)

#define fcl_inference_memory_iho(inputs, layers, hidden, outputs) \
    fcl_base_memory_iho(inputs, layers, hidden, outputs);         \
    fp_t state[2][hidden]

#define fcl_inference_memory_io(inputs, outputs) \
    fcl_base_memory_io(inputs, outputs)

#define fcl_inference_memory_size(inputs, layers, hidden, outputs) ( \
    fcl_base_memory_size(inputs, layers, hidden, outputs) +          \
    fcl_inference_state_count(hidden) * sizeof(fp_t) +               \
    sizeof(fcl_t)                                                    \
)

#define fcl_training_memory_iho(inputs, layers, hidden, outputs) \
    fcl_base_memory_iho(inputs, layers, hidden, outputs);        \
    fp_t state[layers][hidden];                                  \
    fp_t delta[layers][hidden];                                  \
    fp_t od[outputs]
//  ^^^ memory layout is important "od" must be imediately after "delta"

#define fcl_training_memory_io(inputs, outputs)    \
    fcl_base_memory_io(inputs, outputs);           \
    fp_t od[outputs]
//  ^^^ memory layout is important "od" must be imediately after "ow"

#define fcl_training_memory_size(inputs, layers, hidden, outputs) ( \
    fcl_base_memory_size(inputs, layers, hidden, outputs) +         \
   (fcl_training_state_count(layers, hidden) +                      \
    fcl_training_delta_count(layers, hidden) +                      \
    /* od: */ (outputs)) * sizeof(fp_t) +                           \
    sizeof(fcl_t)                                                   \
)

#define fcl_io_memory_size(inputs, layers, hidden, outputs) (       \
    fcl_base_memory_size(inputs, layers, hidden, outputs) +         \
    fcl_inference_state_count(hidden) * sizeof(fp_t) +              \
    (uint8_t*)&((fcl_t*)(0))->iw - (uint8_t*)((fcl_t*)(0))          \
)

#endif // fcl_H

#ifdef FCL_IMPLEMENTATION

static fp_t fcl_random(fcl_t* nn, fp_t weight_range) {
    double r = crt.random64(&nn->seed) / (double)UINT64_MAX * 2 * weight_range - weight_range;
    return (fp_t)r;
}

static void fcl_randomize(fcl_t* nn, fp_t weight_range) {
    for (int i = 0; i < fcl_iw_count(nn->inputs, nn->hidden); i++) {
        nn->iw[i] = fcl.random(nn, weight_range);
    }
    for (int i = 0; i < fcl_hw_count(nn->layers, nn->hidden); i++) {
        nn->hw[i] = fcl.random(nn, weight_range);
    }
    for (int i = 0; i < fcl_ow_count(nn->inputs, nn->hidden, nn->outputs); i++) {
        nn->ow[i] = fcl.random(nn, weight_range);
    }
}

static inline void fcl_check_sizes_and_structs_correctness() {
    // C99 does not have constexpr thus just generic check for a match
    {
        enum { inputs = 123, layers = 234, hidden = 456, outputs = 567 };
        typedef struct {
            fcl_weights_memory_iho(inputs, layers, hidden, outputs);
        } fcl_weights_memory_iho_t;
        static_assert(fcl_weight_memory_size(inputs, layers, hidden, outputs) ==
            sizeof(fcl_weights_memory_iho_t));
        typedef struct {
            fcl_inference_memory_iho(inputs, layers, hidden, outputs);
        } fcl_inference_memory_t;
        static_assert(fcl_inference_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_inference_memory_t) + sizeof(fcl_t));
        typedef struct {
            fcl_training_memory_iho(inputs, layers, hidden, outputs);
        } fcl_training_memory_t;
        static_assert(fcl_training_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_training_memory_t) + sizeof(fcl_t));
    }
    // Microsoft implementation of C99 and C17 does not allow zero size arrays
    {
        enum { inputs = 123, layers = 0, hidden = 0, outputs = 567 };
        typedef struct {
            fcl_weights_memory_io(inputs, outputs);
        } fcl_weights_memory_io_t;
        static_assert(fcl_weight_memory_size(inputs, layers, hidden, outputs) ==
            sizeof(fcl_weights_memory_io_t));
        typedef struct {
            fcl_inference_memory_io(inputs, outputs);
        } fcl_inference_memory_t;
        static_assert(fcl_inference_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_inference_memory_t) + sizeof(fcl_t));
        typedef struct {
            fcl_training_memory_io(inputs, outputs);
        } fcl_training_memory_io_t;
        static_assert(fcl_training_memory_size(inputs, layers, hidden, outputs) ==
                  sizeof(fcl_training_memory_io_t) + sizeof(fcl_t));
    }
}

static void fcl_init(fcl_t* nn, int64_t bytes, uint64_t seed,
        int64_t inputs, int64_t layers, int64_t hidden, int64_t outputs) {
    fcl_check_sizes_and_structs_correctness();
    assert(inputs >= 1);
    assert(layers >= 0);
    assert(layers > 0 && hidden > 0 || layers == 0 && hidden == 0);
    assert(outputs >= 1);
    const int64_t training_size = fcl_training_memory_size(inputs, layers, hidden, outputs);
    const int64_t inference_size = fcl_inference_memory_size(inputs, layers, hidden, outputs);
    uint8_t* p = (uint8_t*)nn;
    memset(p, 0, sizeof(fcl_t)); // only zero out the header
    nn->inputs = inputs;
    nn->layers = layers;
    nn->hidden = hidden;
    nn->outputs = outputs;
    // iw, hw can be huge and deliberately left uninitialized
    nn->iw = (fp_t*)(p + sizeof(fcl_t));
    nn->hw = nn->iw + fcl_iw_count(inputs, hidden);
    nn->ow = nn->hw + fcl_hw_count(layers, hidden);
    nn->output = nn->ow + fcl_ow_count(inputs, hidden, outputs);
    uint8_t* b = p + sizeof(fcl_t);  // beginning of counters memory
    uint8_t* e = p + bytes; // end of memory
//  traceln("total weights=%lld", (fp_t*)e - (fp_t*)b);
//  traceln("");
//  traceln("fcl_iw_count(inputs:%lld hidden:%lld)=%lld", inputs, hidden, fcl_iw_count(inputs, hidden));
//  traceln("fcl_hw_count(layers:%lld hidden:%lld)=%lld", layers, hidden, fcl_hw_count(layers, hidden));
//  traceln("fcl_ow_count(inputs:%lld hidden:%lld outputs:%lld)=%lld", inputs, hidden, outputs, fcl_ow_count(inputs, hidden, outputs));
//  traceln("fcl_weights_count()=%lld", fcl_weights_count(inputs, layers, hidden, outputs));
//  traceln("");
//  traceln("fcl_output_count()=%lld", outputs);
    nn->state = nn->output + outputs; // .output[outputs]
//  traceln("iw - memory %lld (bytes) sizeof(fcl_t)=%lld", (uint8_t*)nn->iw - p, sizeof(fcl_t));
//  traceln("hw - iw %lld", nn->hw - nn->iw);
//  traceln("ow - hw %lld", nn->ow - nn->hw);
//  traceln("output - ow %lld", nn->output - nn->ow);
//  traceln("state - output %lld", nn->state - nn->output);
    assert((uint8_t*)nn->iw - p == sizeof(fcl_t));
    assert(p + sizeof(fcl_t) == (uint8_t*)nn->iw);
    if (bytes == inference_size) {
//      traceln("fcl_inference_state_count()=%lld", fcl_inference_state_count(hidden));
        nn->delta = null;
        nn->od = null;
        assert((uint8_t*)(nn->state + fcl_inference_state_count(hidden)) == e);
    } else if (bytes == training_size) {
        nn->delta = nn->state + fcl_training_state_count(layers, hidden);
        nn->od = nn->delta + fcl_training_delta_count(layers, hidden);
//      traceln("fcl_training_state_count(layers, hidden): %d", fcl_training_state_count(layers, hidden));
//      traceln("fcl_training_delta_count(layers, hidden): %d", fcl_training_delta_count(layers, hidden));
//      traceln("end - nn->state %lld bytes %lld counts",
//           e - (uint8_t*)(nn->od + outputs),
//          (e - (uint8_t*)(nn->od + outputs)) / sizeof(fp_t));
//      traceln("delta - state %lld", nn->delta - nn->state);
//      traceln("od - delta %lld", nn->od - nn->delta);
//      traceln("end - od %lld (bytes) %lld count", e - (uint8_t*)nn->od, (fp_t*)e - nn->od);
        assert((uint8_t*)(nn->od + outputs) == e);
    } else {
        assert(false, "use fcl_infrence|training_memory_size()");
    }
    nn->seed = seed;
    nn->activation_hidden = null;
    nn->derivative_hidden = null;
    nn->activation_output = null;
    nn->derivative_output = null;
    if (nn->layers == 0) {
        nn->iw = null;
        nn->hw = null;
    }
    (void)b; (void)e;
}

static fp_t fcl_sigmoid(fp_t x) {
    assert(!isnan(x));
    fp_t r = (fp_t)(1.0 / (1.0 + exp(-x)));
    assert(!isnan(r));
    return r;
}

static fp_t fcl_sigmoid_derivative(fp_t x) {
    // https://hausetutorials.netlify.app/posts/2019-12-01-neural-networks-deriving-the-sigmoid-derivative/
    fp_t sx = fcl_sigmoid(x); return sx * (1 - sx);
}


static fp_t fcl_relu(fp_t x) {
    return x > 0 ? x : 0;
}

static fp_t fcl_relu_derivative(fp_t x) {
    // https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
//  assert(x != 0, "ReLU derivative undefined for 0"); // TODO: ???
    return (fp_t)(x > 0 ? 1 : 0);
}

static fp_t fcl_tanh(fp_t x) {
    return (fp_t)tanh(x);
}

static fp_t fcl_tanh_derivative(fp_t x) {
    fp_t tx = tanhf(x);
    return (fp_t)(1 - tx * tx);
}

static fp_t fcl_linear(fp_t x) {
    return x;
}

static fp_t fcl_linear_derivative(fp_t x) {
    return 1; (void)x; // unused
}

static fp_t fcl_leaky_relu(fp_t x) { return x > 0 ? x : 0.01f * x; }

static fp_t fcl_leaky_relu_derivative(fp_t x) { return x > 0 ? 1 : 0.01f; }

static fp_t fcl_elu(fp_t x) { return (fp_t)(x >= 0 ? x : (fp_t)(0.01 * (exp(x) - 1))); }
static fp_t fcl_elu_derivative(fp_t x) { return (fp_t)(x >= 0 ? 1 : (fp_t)(0.01 * exp(x))); }

static fp_t fcl_swish(fp_t x) { return x * fcl_sigmoid(x); }
static fp_t fcl_swish_derivative(fp_t x) { fp_t sx = fcl_sigmoid(x); return sx + x * sx * (1 - sx); }

static fcl_activation_t fcl_derivative_of(fcl_activation_t af) {
    if (af == fcl.sigmoid) {
        return fcl_sigmoid_derivative;
    } else if (af == fcl.tanh) {
        return fcl_tanh_derivative;
    } else if (af == fcl.relu) {
        return fcl_relu_derivative;
    } else if (af == fcl.linear) {
        return fcl_linear_derivative;
    } else if (af == fcl.leaky_relu) {
        return fcl_leaky_relu_derivative;
    } else if (af == fcl.elu) {
        return fcl_elu_derivative;
    } else if (af == fcl.swish) {
        return fcl_swish_derivative;
    } else {
        assert(false);
        return NULL;
    }
}

// fcl_dot_product_w_x_i(w[n + 1], i[n], n) treats first element of w as bias

#ifdef DEBUG
#define check_poison(p) fatal_if(*(uint32_t*)p == 0xCDCDCDCDU)
#else
#define check_poison(p)
#endif

static inline fp_t fcl_w_x_i(const fp_t* restrict w,
    const fp_t* restrict i, int64_t n) {
    assert(n > 0);
    check_poison(w);
    fp_t sum = (fp_t)(*w++ * -1.0); // bias
    const fp_t* e = i + n;
    while (i < e) { check_poison(w); check_poison(i); sum += *w++ * *i++; }
    return sum;
}

static fp_t* fcl_inference(fcl_t* nn, const fp_t* input) {
    nn->derivative_hidden = fcl_derivative_of(nn->activation_hidden);
    nn->derivative_output = fcl_derivative_of(nn->activation_output);
    // TODO: delete PARANOIDAL asserts
    assert(fcl_inference_state_count(nn->hidden) == 2 * nn->hidden);
    fp_t* i = (fp_t*)input; // because i/o swap below
    fp_t* s[2] = { nn->state, nn->state + nn->hidden };
    fp_t* o = s[0];
    if (nn->layers > 0) {
        /* input -> hidden layer */
        const fp_t* w = nn->iw;
        for (int64_t j = 0; j < nn->hidden; j++) {
//          traceln("w[0] %.16e", w[0]);
//          for (int64_t k = 0; k < nn->inputs; ++k) {
//              traceln("w[%lld] %.16e i[%lld]: %.16e", k + 1, w[k + 1], k, i[k]);
//          }
            *o++ = nn->activation_hidden(fcl_w_x_i(w, i, nn->inputs));
//          traceln("o[0][%lld]: %.16e", j, *(o - 1));
            w += nn->inputs + 1;
        }
        int ix = 0; // state index for inference only networks
        i = s[0]; // "o" already incremented above
        w = nn->layers > 0 ? nn->hw : nn->iw;
        for (int64_t h = 0; h < nn->layers - 1; h++) {
            for (int64_t j = 0; j < nn->hidden; j++) {
                *o++ = nn->activation_hidden(fcl_w_x_i(w, i, nn->hidden));
//              traceln("o[%lld][%lld]: %.16e", h, j, *(o - 1));
                w += nn->hidden + 1;
            }
            if (nn->delta == null) { // inference only network
                ix = !ix;
                i = s[ix];
                o = s[!ix];
            } else {
                i += nn->hidden; // "o" already incremented above
            }
        }
        if (nn->delta != null) { // training network
            assert(o == nn->state + fcl_training_state_count(nn->layers, nn->hidden));
        }
    }
    // "n" number of output connections from "input" vector
    // if number of hidden layers is zero input is connected to output:
    const int64_t n = nn->layers > 0 ? nn->hidden : nn->inputs;
    const fp_t* w = nn->ow;
    o = nn->output;
    for (int64_t j = 0; j < nn->outputs; j++) {
//      traceln("ouput[%d] w[0]: %.16e", j, *w);
//      for (int64_t k = 0; k < n; ++k) {
//          traceln("ouput[%lld] w[%lld]: %.16e i[%lld]: %.16e", j, k + 1, w[k + 1], k, i[k]);
//      }
        *o++ = nn->activation_output(fcl_w_x_i(w, i, n));
//      traceln("ouput[%lld]: %.16e", j, *(o - 1));
        w += n + 1;
    }
    // TODO: delete this PARANOIDAL assert
    assert((nn->layers > 0 ? w - nn->iw : w - nn->ow) ==
        fcl_weights_count(nn->inputs, nn->layers, nn->hidden, nn->outputs));
    return nn->output;
}

/*
  MSE/L2 loss vs RMSE
  Using the square root of the sum of squares of errors
  (Root Mean Squared Error, RMSE)
  could be another valid option for measuring the loss,
  and it is also commonly used in practice.However, the
  RMSE has some differences and considerations:

  The RMSE will emphasize larger errors more than the MSE,
  which could be beneficial in some cases, especially if
  you want to penalize larger errors more heavily.

  The RMSE is more sensitive to outliers since it involves
  taking the square root of the squared errors.If you have
  outliers in your dataset, they can significantly impact the RMSE.

  The square root operation can be computationally more expensive
  than the simple squared difference.
*/

static fp_t fcl_train(fcl_t* nn, fp_t const* inputs,
        const fp_t* truth, fp_t learning_rate) {
    fp_t loss = 0;
    fp_t const* output = fcl.inference(nn, inputs);
    {   // calculate the output layer deltas.
        fp_t const *o = output; /* First output. */
        fp_t *d = nn->od; // output delta
        fp_t const* t = truth; // pointer to the first grownd truth value
        for (int64_t j = 0; j < nn->outputs; j++) {
            // Mean Squared Error (MSE) or L2 loss
            loss += (fp_t)(0.5 * (*t - *o) * (*t - *o));
            *d++ = (*t - *o) * nn->derivative_output(*o);
//          traceln("o[%lld]: %.17e od[%lld]: %.17e", j, *o, j, *(d - 1));
            o++; t++;
        }
    }
    assert(nn->delta + nn->layers * nn->hidden == nn->od);
    // hidden layer deltas, start at the last layer and work upward.
    const int64_t hh1 = (nn->hidden + 1) * nn->hidden;
    fp_t* ww = nn->ow;
    assert(nn->output + nn->outputs == nn->state);
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        if (h != nn->layers) {
            assert(ww == nn->hw + hh1 * h);
        }
        /* Find first output and delta in this layer. */
        fp_t *o = nn->state + nn->hidden * h;
        fp_t *d = nn->delta + nn->hidden * h;
        /* Find first delta in following layer (which may be .delta[] or .od[]). */
        fp_t const * const dd = (h == nn->layers - 1) ?
            nn->od : nn->delta + (h + 1) * nn->hidden;
        if (h == nn->layers - 1) {
            assert(dd == nn->od);
        }
        for (int64_t j = 0; j < nn->hidden; j++) {
            fp_t delta = 0;
            const int64_t n = h == nn->layers - 1 ? nn->outputs : nn->hidden;
            for (int64_t k = 0; k < n; k++) {
                const fp_t forward_delta = dd[k];
//              traceln("dd[%lld]: %25.17e", k, dd[k]);
                const int64_t windex = k * (nn->hidden + 1) + (j + 1);
                const fp_t forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
//              traceln("delta: %25.17e := forward_delta: %25.17e "
//                      "forward_weight: %25.17e\n", delta, forward_delta,
//                      forward_weight);
            }
            *d = nn->derivative_hidden(*o) * delta;
//          traceln("d[%lld]: %25.17e o: %25.17e delta: %25.17e", j, *d, *o, delta);
            d++; o++;
        }
        ww = nn->hw + (h - 1) * hh1;
//      traceln("nn->hw - ww %lld", nn->hw - ww);
    }
    {   // Train the outputs.
        fp_t const *d = nn->od; /* output delta. */
        /* Find first weight to first output delta. */
        fp_t *w = nn->ow;
        /* Find first output in previous layer. */
        fp_t const * const i =
            nn->layers == 0 ? inputs :
                              nn->state + nn->hidden * (nn->layers - 1);
        const int64_t n = nn->layers == 0 ? nn->inputs : nn->hidden;
        /* Set output layer weights. */
        for (int64_t j = 0; j < nn->outputs; j++) {
            *w++ += (fp_t)(*d * learning_rate * -1.0); // bias
            for (int64_t k = 1; k < n + 1; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              traceln("output[%lld] i[%lld] %25.17e w %25.17e",
//                  j, k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
        assert((nn->layers > 0 ? w - nn->iw : w - nn->ow) ==
            fcl_weights_count(nn->inputs, nn->layers, nn->hidden, nn->outputs));
    }
    /* Train the hidden layers. */
    for (int64_t h = nn->layers - 1; h >= 0; h--) {
        fp_t const* d = nn->delta + h * nn->hidden;
        fp_t const* i = h == 0 ? inputs : nn->state + (h - 1) * nn->hidden;
        fp_t *w = h == 0 ? nn->iw : nn->hw + hh1 * (h - 1);
        for (int64_t j = 0; j < nn->hidden; j++) {
//          traceln("hidden layer[%lld][%lld] weights w.ofs=%lld\n", h, j, w - nn->iw);
            *w++ += (fp_t)(*d * learning_rate * -1.0); // bias
//          traceln("w[0] (bias)=%25.17e d=%25.17e", *(w - 1), *d);
            const int64_t n = (h == 0 ? nn->inputs : nn->hidden) + 1;
            for (int64_t k = 1; k < n; k++) {
                *w++ += *d * learning_rate * i[k - 1];
//              traceln("i[%lld] %25.17e w %25.17e ", k - 1, i[k - 1], *(w - 1));
            }
            ++d;
        }
    }
    return loss;
}


fcl_if fcl = {
    .init = fcl_init,
    .random = fcl_random,
    .randomize = fcl_randomize,
    .inference = fcl_inference,
    .train = fcl_train,
    .sigmoid = fcl_sigmoid,
    .relu = fcl_relu,
    .tanh = fcl_tanh,
    .linear = fcl_linear,
    .leaky_relu = fcl_leaky_relu,
    .elu = fcl_elu,
    .swish = fcl_swish
};

#endif // FCL_IMPLEMENTATION

#ifdef __cplusplus
}
#endif


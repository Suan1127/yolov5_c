#include "activation.h"
#include <math.h>

void activation_silu(tensor_t* t) {
    if (!t || !t->data) return;
    
    size_t size = tensor_size(t);
    for (size_t i = 0; i < size; i++) {
        float x = t->data[i];
        t->data[i] = x * sigmoid(x);
    }
}

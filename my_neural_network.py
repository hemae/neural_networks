import numpy as np
import random


def linear(x, a=1):
    return a * x


def der_linear(x, a=1):
    return a


def sigmoida(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoida(x):
    return sigmoida(x) * (1 - sigmoida(x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def der_tanh(x):
    return 1 - (tanh(x)) * (tanh(x))


def relu(x, a=1):
    if x < a:
        return 0
    else:
        return x


def der_relu(x, a=1):
    if x < a:
        return 0
    else:
        return 1


funcs = {
    'one': {
        'linear': linear,
        'sigmoida': sigmoida,
        'tanh': tanh,
        'relu': relu
    },
    'der': {
        'linear': der_linear,
        'sigmoida': der_sigmoida,
        'tanh': der_tanh,
        'relu': der_relu
    }
}


def neuron(input_signal, activation_func=None):
    if activation_func == None:
        return input_signal
    else:
        return funcs['one'][activation_func](input_signal)


def layer(neurons_number, input_vector, weight_vector, activation_func=None):
    output_vector = []
    for_total_input_vector = []
    for i in range(neurons_number):
        input_val = np.dot(weight_vector[i], input_vector)
        for_total_input_vector.append(input_val)
        neuron_output_val = neuron(input_val, activation_func=activation_func)
        output_vector.append(neuron_output_val)

    return np.array(output_vector), np.array(for_total_input_vector)


def network(network_scheme, input_vector, total_weight_vector, activation_func=None, activation_func_output=None):
    vector_data = input_vector
    total_input_vector = []
    total_output_vector = []
    total_input_vector.append(np.array(input_vector))
    total_output_vector.append(np.array(input_vector))
    for i in range(1, len(network_scheme) - 1):
        vector_data, for_total_input_vector = layer(network_scheme[i], vector_data, total_weight_vector[i - 1],
                                                    activation_func=activation_func)
        total_input_vector.append(for_total_input_vector)
        total_output_vector.append(vector_data)
    vector_data, for_total_input_vector = layer(network_scheme[-1], vector_data, total_weight_vector[-1],
                                                activation_func=activation_func_output)
    total_input_vector.append(for_total_input_vector)
    total_output_vector.append(vector_data)

    return vector_data, total_input_vector, total_output_vector


def initialize_total_weight_vector(network_scheme):
    total_weight_vector = []
    for i in range(1, len(network_scheme)):
        layer_weight_vector = []
        for j in range(network_scheme[i]):
            neuron_weight_vector = np.random.rand(network_scheme[i - 1]) - 0.5
            layer_weight_vector.append(neuron_weight_vector)
        total_weight_vector.append(np.array(layer_weight_vector))

    return total_weight_vector


def back_propagation_iteration(total_weight_vector, total_input_vector, total_output_vector, error, conv_step,
                               activation_func, activation_func_output):
    grad_vector = error * funcs['der'][activation_func_output](total_input_vector[-1])
    for i in reversed(range(len(total_weight_vector))):
        for j in range(len(grad_vector)):
            new_grad_vector = np.zeros(len(total_output_vector[i]), dtype=float)
            total_weight_vector[i][j] = total_weight_vector[i][j] - conv_step * grad_vector[j] * total_output_vector[i]
            new_grad_vector += funcs['der'][activation_func](total_input_vector[i]) * total_weight_vector[i][j] * \
                               grad_vector[j]
        grad_vector = new_grad_vector

    return total_weight_vector


network_scheme = [3, 2, 1]
conv_step = 0.1
N = 10000

input_data = [
    [[0, 0, 0], [0]],
    [[0, 0, 1], [1]],
    [[0, 1, 0], [0]],
    [[0, 1, 1], [1]],
    [[1, 0, 0], [0]],
    [[1, 0, 1], [1]],
    [[1, 1, 0], [0]],
    [[1, 1, 1], [0]]
]
total_weight_vector = initialize_total_weight_vector(network_scheme)

for i in range(N):
    k = random.randint(0, 7)
    input_vector = np.array(input_data[k][0])
    output_vector = np.array(input_data[k][1])
    forward_propagation_res, total_input_vector, total_output_vector = network(network_scheme, input_vector,
                                                                               total_weight_vector,
                                                                               activation_func='linear',
                                                                               activation_func_output='linear')
    error = forward_propagation_res - output_vector
    total_weight_vector = back_propagation_iteration(total_weight_vector, total_input_vector, total_output_vector, error,
                                                     conv_step, activation_func='linear', activation_func_output='linear')

for i in range(len(input_data)):
    input_vector = np.array(input_data[i][0])
    output_vector = np.array(input_data[i][1])
    forward_propagation_res, total_input_vector, total_output_vector = network(network_scheme, input_vector,
                                                                               total_weight_vector,
                                                                               activation_func='linear',
                                                                               activation_func_output='linear')
    print(forward_propagation_res)
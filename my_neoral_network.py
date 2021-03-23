import numpy as np


def relu(x, a=0):
    if x <= a:
        return 0
    else:
        return x

def neuron(input_signal, activation_func=None):
    if activation_func == None:
        return input_signal
    else:
        return activation_func(input_signal)

def layer(neurons_number, input_vector, weight_vector, activation_func=None):
    output_vector = []
    for i in range(neurons_number):
        input_val = np.dot(weight_vector[i], input_vector)
        neuron_output_val = neuron(input_val, activation_func=activation_func)
        output_vector.append(neuron_output_val)

    return np.array(output_vector)

def network(network_scheme, input_vector, total_weight_vector, activation_func=None, activation_func_output=None):
    vector_data = input_vector
    for i in range(1, len(network_scheme) - 1):
        vector_data = layer(network_scheme[i], vector_data, total_weight_vector[i - 1], activation_func=activation_func)
    vector_data = layer(network_scheme[-1], vector_data, total_weight_vector[-1], activation_func=activation_func_output)

    return vector_data

def initialize_total_weight_vector(network_scheme):
    total_weight_vector = []
    for i in range(1, len(network_scheme)):
        layer_weight_vector = []
        for j in range(network_scheme[i]):
            neuron_weight_vector = np.random.rand(network_scheme[i - 1]) - 0.5
            layer_weight_vector.append(neuron_weight_vector)
        total_weight_vector.append(np.array(layer_weight_vector))

    return total_weight_vector


network_scheme = [4, 5, 100, 2]
input_vector = [-1, 1, -1, 1]
total_weight_vector = initialize_total_weight_vector(network_scheme)
# print(total_weight_vector)

forward_propogation = network(network_scheme, input_vector, total_weight_vector)
print(forward_propogation)


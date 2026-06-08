import numpy as np
def relu(input):

    output = max(0, input)
    return output


def forward_pass(inputs: list, weights1: list, weights2: list, weights3: list) -> float:
    input_data = np.array(inputs)

    weights_data = {
        'node_0': np.array([weights1]),
        'node_1': np.array([weights2]),
        'output_node': np.array([weights3])
    }


    node_0_input = (input_data * weights_data['node_0']).sum()
    node_0_output = relu(node_0_input)

    node_1_input = (input_data * weights_data['node_1']).sum()
    node_1_output = relu(node_1_input)

    hidden_layer_output = np.array([node_0_output, node_1_output])

    output = (hidden_layer_output * weights_data['output_node']).sum()

    return output 

forward = forward_pass([2, 3], [1, 1], [-1, 1], [2, -1])
print(forward)
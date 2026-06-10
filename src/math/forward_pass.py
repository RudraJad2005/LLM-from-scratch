import numpy as np

def forward_pass(inputs: np.ndarray, weights1: np.ndarray, weights2: np.ndarray, weights3: np.ndarray) -> np.ndarray:
    input_data = np.array(inputs)

    weights_data = {
        'node_0': np.array([weights1]),
        'node_1': np.array([weights2]),
        'output_node': np.array([weights3])
    }

    node_0_value = (input_data * weights_data['node_0']).sum()
    node_1_value = (input_data * weights_data['node_1']).sum()

    hidden_layer_output = np.array([node_0_value, node_1_value])

    output = (hidden_layer_output * weights_data['output_node']).sum()
    return output

forward = forward_pass([2, 3], [1, 1], [-1, 1], [2, -1])
print(forward)


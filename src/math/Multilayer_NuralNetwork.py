import numpy as np

def relu(input):

    output = max(0, input)
    return output

class MultiLayerNeuralNetwork:
    def __init__(self, weights1: list, weights2: list, weights3: list, weights4: list, weights5: list):
        self.weights1 = weights1
        self.weights2 = weights2
        self.weights3 = weights3
        self.weights4 = weights4
        self.weights5 = weights5

    def forward_pass(self, inputs: list) -> float:
        input_data = np.array(inputs)

        weights_data = {
            'node_0_0': np.array([self.weights1]),
            'node_0_1': np.array([self.weights2]),
            'node_1_0': np.array([self.weights3]),
            'node_1_1': np.array([self.weights4]),
            'output_node': np.array([self.weights5])
        }

        return input_data, weights_data

    def predict_with_network(self, input_data):
        
        node_0_0_input = (input_data * self.weights1).sum()
        node_0_0_output = relu(node_0_0_input)

        node_0_1_input = (input_data * self.weights2).sum()
        node_0_1_output = relu(node_0_1_input)

        hidden_0_output = np.array([node_0_0_output, node_0_1_output])

        node_1_0_input = (hidden_0_output * self.weights3).sum()
        node_1_0_output = relu(node_1_0_input)

        node_1_1_input = (hidden_0_output * self.weights4).sum()
        node_1_1_output = relu(node_1_1_input)

        hidden_1_output = np.array([node_1_0_output, node_1_1_output])

        output = (hidden_1_output * self.weights5).sum()

        return(output)


nn = MultiLayerNeuralNetwork([1, 1], [-1, 1], [2, -1], [1, 2], [1, -1])
input_data, weights_data = nn.forward_pass([2, 3])

nn.weights1 = weights_data['node_0_0']
nn.weights2 = weights_data['node_0_1']
nn.weights3 = weights_data['node_1_0']
nn.weights4 = weights_data['node_1_1']
nn.weights5 = weights_data['output_node']
prediction = nn.predict_with_network(input_data)
print(prediction)
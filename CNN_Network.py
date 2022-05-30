def init_network():
    network = {}
    network['W1'] = np.array([[w11,w13,w15],[w12,w14,w16]])
    network['W2'] = np.array([[w21,w24],[w22,w25],[w23,w26]])
    network['W3'] = np.array([[w31,w33],[w32,w34]])
    network['b1'] = np.array([b11,b12,b13])
    network['b2'] = np.array([b21,b22,b23])
    network['b3'] = np.array([b31,b32,b33])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    B1, B2, B3 = network['B1'],network['B2'],network['B3']
    a1 = np.dot(x, W1) + B1
    a1 = sigmoid(a1)
    a2 = np.dot(a1, W2) + B2
    a2 = sigmoid(a2)
    a3 = np.dot(a2, W3) + B3
    a3 = sigmoid(a3)
    return a3

if __name__ = "__main__":
    network = init_network()
    input_vector = [input() for _ in range(2)]
    output_vector = forward(network,input_vector)
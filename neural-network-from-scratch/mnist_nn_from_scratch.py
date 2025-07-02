import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

np.random.seed(0)

# ---------- Utility Functions ----------

def load_data():
    X = np.load('E:/kolya/ML Course/Videos/9 Multiclase/mnist-sample/mnist-sample/X.npy')
    y = np.load('E:/kolya/ML Course/Videos/9 Multiclase/mnist-sample/mnist-sample/y.npy')

    # Normalize
    X = X / 255.0
    return X, y

def dtanh(y):
    return 1 - y ** 2

def softmax_batch(x):
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_batch(y_true, y_pred): 
    cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-15), axis=1)
    return np.mean(cross_entropy)

# ---------- Generalized Neural Network Class ----------

class GeneralNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.activation = np.tanh
        self.dactivation = dtanh
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]  
            fan_out = layer_sizes[i + 1]
            W = np.random.randn(fan_in, fan_out)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
            # print(f"\nLayer {i+1} - W.shape: {W.shape}, b.shape: {b.shape}")
            # print("W sample:\n", W[:2, :3])  # Show first 2 rows × 3 columns
            # print("b sample:\n", b)            

    def forward(self, X):
        activations = [X] 
        nets = []

        for i in range(len(self.weights)):
            
            net = activations[-1] @ self.weights[i] + self.biases[i]  
            nets.append(net)

            if i == len(self.weights) - 1:
                # Output layer uses softmax
                out = softmax_batch(net)
            else:
                out = self.activation(net)
            activations.append(out)

        return activations, nets

    def backward(self, y_batch, activations, nets):
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Output layer gradient
        delta = activations[-1] - y_batch

        for i in reversed(range(len(self.weights))):
            grads_W[i] = activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i != 0:  # Not the input layer then update delta for next layer 
                dE_dnet = delta @ self.weights[i].T
                dE_dout = dE_dnet * self.dactivation(activations[i])
                delta = dE_dout

        return grads_W, grads_b

    def update(self, grads_W, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, X_train, y_train, X_test, y_test, epochs=20, lr=1e-2, batch_size=32):
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])  
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                activations, nets = self.forward(X_batch)
                grads_W, grads_b = self.backward(y_batch, activations, nets)
                self.update(grads_W, grads_b, lr)

            # After epoch: compute loss & accuracy on test set
            out_test = self.forward(X_test)[0][-1] # the last output layer
            loss = cross_entropy_batch(y_test, out_test)
            acc = self.evaluate(X_test, y_test)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def evaluate(self, X, y_true):
        y_pred = np.argmax(self.forward(X)[0][-1], axis=1)
        y_true = np.argmax(y_true, axis=1)
        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    X, y = load_data()
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Define architecture: input(features), hidden1, hidden2, output
    layer_sizes = [X_train.shape[1], 20, 15, 10]
    model = GeneralNeuralNetwork(layer_sizes)

    # Train the model
    model.train(X_train, y_train, X_test, y_test, epochs=20, lr=1e-2, batch_size=32)

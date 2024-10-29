import numpy as np

# Initialize weights (1 set for this example)
weights = np.random.rand(1, 2)  # Random initialization for weights

# Hardcoded inputs for the perceptron
inputs = np.array([
    [1, 2], [2, 3], [3, 1], [4, 2], [5, 3],
    [6, 1], [7, 2], [8, 3], [1, 1], [2, 2],
    [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
    [8, 8], [9, 1], [1, 9], [3, 4], [4, 3],
    [2, 5], [5, 2], [6, 4], [4, 6], [2, 1],
    [1, 2], [3, 2], [2, 3], [5, 5], [1, 6],
    [6, 1], [2, 7], [7, 2], [8, 1], [9, 3],
    [3, 6], [6, 3], [4, 4], [5, 5], [2, 4],
    [4, 2], [1, 8], [8, 1], [5, 4], [4, 5],
    [2, 6], [6, 2], [3, 5], [5, 3], [2, 2],
    [7, 3], [3, 7], [1, 3], [3, 1], [2, 8],
    [8, 2], [6, 5], [5, 6], [1, 1], [2, 2],
    [3, 3], [4, 4], [5, 5], [6, 6], [7, 7],
    [8, 8], [9, 9], [1, 4], [4, 1], [2, 5],
    [5, 2], [3, 6], [6, 3], [2, 4], [4, 2],
    [1, 7], [7, 1], [3, 5], [5, 3], [6, 4],
    [4, 6], [2, 3], [3, 2], [1, 8], [8, 1],
    [3, 3], [6, 2], [2, 6], [4, 5], [5, 4],
    [1, 9], [9, 1], [2, 8], [8, 2], [7, 3],
    [3, 7], [1, 5], [5, 1], [2, 9], [9, 2],
])

# Define outputs for each of the inputs (Example binary outputs)
outputs = np.random.randint(0, 2, size=(inputs.shape[0],))  # 100 outputs matching inputs

def sig(x):
    return 1 / (1 + np.exp(-x))

# Initial bias and learning rate
bias = np.random.rand()  # Random initialization for bias
lr = 0.01  # Set a small learning rate
num_epochs = 100  # Number of training epochs

# Training loop
for epoch in range(num_epochs):
    for i, d in zip(inputs, outputs):
        sum_result = np.dot(i, weights[0]) + bias
        r = sig(sum_result)  # Sigmoid output (prediction)
        error = d - r  # Calculate error

        # Update weights based on error
        weights[0][0] += lr * error * i[0]  # Update first weight
        weights[0][1] += lr * error * i[1]  # Update second weight

        bias += lr * error  # Update the bias

        # Print the results for this iteration (optional)
        print(f"Epoch: {epoch}, Input: {i}, Weighted Sum + Bias: {sum_result:.4f}, "
              f"Sigmoid Output: {r:.4f}, Target: {d}, Error: {error:.4f}")

# Final weights and bias after training
print("Final Weights:", weights)
print("Final Bias:", bias)


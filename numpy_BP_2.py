import numpy as np
from keras import datasets
from keras import utils

(X_train, y_train), (test_X, test_y) = datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784) / 255.0
test_X = test_X.reshape(test_X.shape[0], 784) / 255.0

input_size = 784
num_hidden = 128
num_classes = 10
num_passes = 10 
batch_size = 60

# 初始化权重
np.random.seed(40)
#W1 = np.random.randn(input_size, num_hidden) * 0.01
#W2 = np.random.randn(num_hidden, num_classes) * 0.01


#He初始化
W1 = np.random.normal(0,np.sqrt(2.0/input_size), (input_size, num_hidden))
W2 = np.random.normal(0,np.sqrt(2.0/num_hidden), (num_hidden, num_classes))
b1 = np.zeros(num_hidden)
b2 = np.zeros(num_classes)
    
learning_rate = 0.01
reg_lamda = 0 
num_examples = X_train.shape[0]

loss_history = []


for i in range(num_passes):
    for batch in range(0, num_examples, batch_size):
        X_batch = X_train[batch:batch + batch_size]
        y_batch = y_train[batch:batch + batch_size]
        z1 = X_batch.dot(W1) + b1
        a1 = np.maximum(0, z1)
        z2 = a1.dot(W2) + b2

        max_z2 = np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(z2 - max_z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


        loss = -np.mean(np.log(probs[range(len(y_batch)), y_batch] + 1e-10))
        loss_history.append(loss)

        delta3 = probs
        delta3[range(len(y_batch)), y_batch] -= 1

        dW2 = np.dot(a1.T, delta3) + reg_lamda * W2
        db2 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, W2.T) * (a1 > 0)
        dW1 = np.dot(X_batch.T, delta2) + reg_lamda * W1
        db1 = np.sum(delta2, axis=0)


        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f'Iteration {i}, Loss: {loss:.4f}')

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.maximum(0, z1)
    z2 = a1.dot(W2) + b2

    max_z2 = np.max(z2, axis=1, keepdims=True)
    exp_scores = np.exp(z2 - max_z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

model = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

test_pred = predict(model, test_X)
accuracy = np.mean(test_pred == test_y)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


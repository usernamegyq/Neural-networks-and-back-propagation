
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


num_hidden = 7 
num_classes = len(np.unique(y))
num_passes = 100

W1 = np.random.rand(X_train.shape[1], num_hidden)
W2 = np.random.rand(num_hidden, num_classes)
b1 = np.zeros(num_hidden)
b2 = np.zeros(num_classes)

learning_rate = 0.03
reg_lamda = 0.005
num_examples = X_train.shape[0]

def compute_loss(y_true, y_pred):
    num_examples = y_true.shape[0]
    log_probs = -np.log(y_pred[range(num_examples), y_true])
    loss = np.sum(log_probs) / num_examples
    return loss

loss_history = []


for i in range(num_passes):
    z1 = X_train.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = -np.mean(np.log(probs[range(num_examples), y_train]))
    loss_history.append(loss)

    delta3 = probs
    delta3[range(num_examples), y_train] -= 1
    dW2 = np.dot(a1.T, delta3)
    db2 = np.sum(delta3, axis=0)
    delta2 = (1 - np.power(a1, 2)) * np.dot(delta3, W2.T)
    dW1 = np.dot(X_train.T, delta2)
    db1 = np.sum(delta2, axis=0)


    dW1 += reg_lamda * W1
    dW2 += reg_lamda * W2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if i % 10 == 0:
        print(f'Iteration {i}, Loss: {loss:.4f}')


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


model = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}


train_result = predict(model, X_train)
test_result = predict(model, X_test)
print(f'Train accuracy: {np.mean(train_result == y_train):.2f}')
print(f'Test accuracy: {np.mean(test_result == y_test):.2f}')


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = predict(model, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title('Decision Boundary for HAI-F Moon Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

plot_decision_boundary(model, X, y)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

class artificial_neuron:
    """
    Class pour créer un neurone artificiel basique
    """
    def __init__(self, n_iter=100, learning_rate=0.1):
        self.coef_ = None
        self.bias_ = None
        self.n_iter_ = n_iter
        self.learning_rate_ = learning_rate
        self.loss_ = []

    def predict_proba(self, X):
        Z = X.dot(self.coef_) + self.bias_
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def predict(self, X):
        A = self.predict_proba(X)
        return A >= 0.5

    def log_loss(self, y, A):
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
    
    def display_loss(self):
        plt.plot(self.loss_)
        plt.show()

    def fit(self, X, y):
        # initialiser coef et bias
        self.coef_ = np.random.randn(X.shape[1], 1)
        self.bias_ = np.random.randn(1)
        self.loss_ = []

        # apprentissage
        for i in range(self.n_iter_):
            # activations
            A = self.predict_proba(X)

            # loss
            self.loss_.append(self.log_loss(y, A))

            # gradients
            dW = 1 / len(y) * np.dot(X.T, A - y)
            db = 1 / len(y) * np.sum(A - y)

            # mettre à jour
            self.coef_ = self.coef_ - self.learning_rate_ * dW
            self.bias_ = self.bias_ - self.learning_rate_ * db

model = artificial_neuron()
model.fit(X, y)
model.display_loss()

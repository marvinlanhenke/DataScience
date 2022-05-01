import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LogisticRegression:
    def __init__(self, learning_rate=1e-3, n_iters=1000):
        # init parameters
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _get_prediction(self, X):
        linear = np.dot(X, self.weights) + self.bias
        # apply sigmoid function
        sigmoid = 1 / (1 + np.exp(-linear))
        return sigmoid
    
    def _init_params(self):
        self.weights = np.zeros(self.n_features)
        self.bias = 0
    
    def _update_params(self, dw, db):
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
    
    def _get_gradients(self, X, y, y_pred):
        # get distance between y_pred and y_true
        error = y_pred - y
        # compute the gradients of weight & bias
        dw = (1 / self.n_samples) * np.dot(X.T, error)
        db = (1 / self.n_samples) * np.sum(error)
        return dw, db
    
    def fit(self, X, y):
        # get number of samples & features
        self.n_samples, self.n_features = X.shape
        # init weights & bias
        self._init_params()

        # perform gradient descent for n iterations
        for _ in range(self.n_iters):
            # get y_prediction
            y_pred = self._get_prediction(X)
            # compute gradients
            dw, db = self._get_gradients(X, y, y_pred)
            # update weights & bias with gradients
            self._update_params(dw, db)
    
    def predict(self, X):
        y_pred = self._get_prediction(X)
        # use decision boundary to map to binary classes
        y_class = [1 if p > 0.5 else 0 for p in y_pred]
        return y_class


if __name__ == "__main__":

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    logReg = LogisticRegression(learning_rate=0.001, n_iters=1000)
    logReg.fit(X_train, y_train)
    predictions = logReg.predict(X_test)

    print("LogReg Accuracy:", accuracy(y_test, predictions))
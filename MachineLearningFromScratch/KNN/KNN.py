import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        # get predictions for every row in test data
        y_pred = [self._get_single_prediction(x_test_row) for x_test_row in X_test]
        return np.array(y_pred)

    def _get_single_prediction(self, x_test_row):
        # get distances of test_row vs all training rows
        distances = [self._get_euclidean_distance(x_test_row, x_train_row) 
            for x_train_row in self.X_train]
        # get indices of k-nearest neighbors -> k-smallest distances
        k_idx = np.argsort(distances)[:self.k]
        # get corresponding y-labels of training data
        k_labels = [self.y_train[idx] for idx in k_idx]
        # return most common label
        return np.argmax(np.bincount(k_labels))

    def _get_euclidean_distance(self, x1, x2):
        # calculate euclidean distance for a row pair
        sum_squared_distance = np.sum((x1 - x2)**2)
        return np.sqrt(sum_squared_distance)

# Testing
if __name__ == "__main__":

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    scores = []
    
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (idx_train, idx_valid) in enumerate(cv.split(X)):
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        k = 3
        clf = KNN(k=k)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_valid)

        score = accuracy(y_valid, predictions)
        scores.append(score)

    print(f"Mean Accuracy: {np.mean(scores)}")
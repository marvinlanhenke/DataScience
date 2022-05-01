import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))

        self.mean = np.zeros((self.n_classes, self.n_features))
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            X_c = X[y == c]

            self.mean[c, :] = np.mean(X_c, axis=0)
            self.variance[c, :] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / self.n_samples

    def predict(self, X):
        y_hat = [self.get_class_probability(x) for x in X]
        return np.array(y_hat)

    def get_class_probability(self, x):
        posteriors = list()

        for c in range(self.n_classes):
            mean = self.mean[c]
            variance = self.variance[c]
            prior = np.log(self.priors[c])

            posterior = np.sum(np.log(self.gaussian_density(x, mean, variance)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return np.argmax(posteriors)

    def gaussian_density(self, x, mean, var):
        const = 1 / np.sqrt(var * 2 * np.pi)
        proba = np.exp(-0.5 * ((x - mean) ** 2 / var))

        return const * proba

# Testing
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    def get_accuracy(y_true, y_hat):
        return np.sum(y_true==y_hat) / len(y_true)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print('Naive Bayes Accuracy: ', get_accuracy(y_test, predictions))
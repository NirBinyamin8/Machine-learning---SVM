from Task4_sec2 import linear_kernel
import itertools
import qpsolvers as qps
import numpy as np

class SVM():
    def __init__(self,kernel=linear_kernel, degree=0, C=0, gamma=0):
        self.kernel=kernel
        self.degree=degree
        self.C=C
        self.gamma=gamma
        self.X_train=None
        self.y_train=None
        self.alpha=None
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
        ker=self.kernel
        d=self.degree
        g=self.gamma
        N = X.shape[0]
        P = np.empty((N, N))
        for i, j in itertools.product(range(N), range(N)):
            if d != 0:
                P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :], degree=d)
            else:
                if g != 0:
                    P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :], gamme=g)
                else:
                    P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :])
        P = 0.5 * (P + P.T)
        P = 0.5 * P
        q = -np.ones(N)
        GG = -np.eye(N)
        h = np.zeros(N)

        self.alpha = qps.solve_qp(P, q, GG, h, solver='osqp')
    def predict(self,X):
        decision_func=self.decision_function(X)
        return np.sign(decision_func)
    def decision_function(self,X):
        d = self.degree
        g = self.gamma
        alpha = self.alpha
        X_train = self.X_train
        X_test = X
        y_train = self.y_train
        kernel = self.kernel
        predictions = []
        for x in X_test:
            if d != 0:
                prediction = np.sum(alpha * y_train * np.array([kernel(x, xi, degree=d) for xi in X_train]))
                predictions.append(prediction)
            else:
                if g != 0:
                    prediction = np.sum(alpha * y_train * np.array([kernel(x, xi, gamme=g) for xi in X_train]))
                    predictions.append(prediction)
                else:
                    prediction = np.sum(alpha * y_train * np.array([kernel(x, xi) for xi in X_train]))
                    predictions.append(prediction)

        return np.array(predictions)

    def score(self,X, y):
        y_pred=self.predict(X)
        correct_predictions = np.sum(y == y_pred)
        total_samples = len(y)
        score = correct_predictions / total_samples
        return score


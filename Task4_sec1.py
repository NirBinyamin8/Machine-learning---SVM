import qpsolvers as qps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def svm_primal(X, y):
    N = X.shape[0]
    n = X.shape[1]+1
    # Add bias
    X = np.c_[X, np.ones(N)]
    # w.t * P *w
    P = np.eye(n)
    # The linear part (no needed)
    q = np.zeros(n)
    # The inequality part
    G = -np.diag(y) @ X
    h = -np.ones(N)
    # Python's quadratic program
    w = qps.solve_qp(P, q, G, h, solver='osqp')
    return w
def svm_dual(X,y):
    N = X.shape[0]
    X = np.c_[X, np.ones(N)]
    G = np.diag(y) @ X
    P = G @ G.T
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)
    alpha = qps.solve_qp(P, q, GG, h, solver='osqp')
    w = G.T @ alpha
    return(w)

def plot_dataset(X, y, w):
    r = np.where(y < 0)
    b = np.where(y > 0)

    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    plt.scatter(X[r, 0], X[r, 1], color='red', label='Class -1')
    plt.scatter(X[b, 0], X[b, 1], color='blue', label='Class 1')

    plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])

    lx = np.linspace(x_min, x_max, 60)

    ly = [(-w[-1] - w[0] * p) / w[1] for p in lx]
    plt.plot(lx, ly, color='black')

    ly1 = [(-w[-1] - w[0] * p - 1) / w[1] for p in lx]
    plt.plot(lx, ly1, "--", color='red')

    ly2 = [(-w[-1] - w[0] * p + 1) / w[1] for p in lx]
    plt.plot(lx, ly2, "--", color='blue')
    plt.show()

if __name__ == "__main__":
    # Load the Data
    path = r"C:\Users\97252\Desktop\שיעורי בית\שנה ב\סמסטר ב\מבוא ללמידת מכונה\תרגיל 4\simple_classification.csv"
    df = pd.read_csv(path)
    #Changing the representation of tags
    y = np.array(df['y'])
    y=y*2 -1
    X=np.array(df.drop(columns=['y']))

    # Primal and plot
    w=svm_primal(X,y)
    print("QP solution of primal program : w = {}".format(w))
    plot_dataset(X, y,w)

    # Dual and plot
    w=svm_dual(X,y)
    print("QP solution of dual program : w = {}".format(w))
    plot_dataset(X, y, w)










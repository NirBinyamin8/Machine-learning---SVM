import qpsolvers as qps
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib
def predict(X_test, X_train, y_train, kernel, alpha, d=0,g=0):
    predictions = []
    for x in X_test:
        if d!=0:
            prediction = np.sign(np.sum(alpha * y_train * np.array([kernel(x, xi,degree=d) for xi in X_train])))
            predictions.append(prediction)
        else:
            if g!=0:
                prediction = np.sign(np.sum(alpha * y_train * np.array([kernel(x, xi,gamme=g) for xi in X_train])))
                predictions.append(prediction)
            else:
                prediction = np.sign(np.sum(alpha * y_train * np.array([kernel(x, xi) for xi in X_train])))
                predictions.append(prediction)


    return np.array(predictions)
def calculate_loss(y_true, y_pred):
    incorrect_predictions = np.sum(y_true != y_pred)
    total_samples = len(y_true)
    loss = incorrect_predictions / total_samples
    return loss
def plot_loss_bar(loss_values,kernel_names):
    # Create a bar plot
    plt.bar(kernel_names, loss_values)
    plt.xlabel('Kernel')
    plt.ylabel('Loss')
    plt.title('Loss for Different Kernels')
    plt.xticks(rotation=20)
    plt.xticks(fontsize=8)

    # Display the plot
    plt.show()




def highlight_support_vectors(X, alpha):
    sv = support_vectors_func(alpha)
    plt.scatter(X[sv,0], X[sv,1], s=300, linewidth=3, facecolors='none', edgecolors='k')
def support_vectors_func(alpha, thresh=0.0001):
    return np.argwhere(np.abs(alpha) > thresh).reshape(-1)
def plot_classifier_z_kernel(alpha, X, y,ker,Name,s=None,d=0,g=0):
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    xx = np.linspace(x_min, x_max)
    yy = np.linspace(y_min, y_max)

    xx, yy = np.meshgrid(xx, yy)

    N = X.shape[0]
    z = np.zeros(xx.shape)
    for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
        if d!=0:
            z[i, j] = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]]),degree=d) for k in range(N)])
        else:
            if g!=0:
                z[i, j] = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]]),gamme=g) for k in range(N)])
            else:
                z[i, j] = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]])) for k in range(N)])





    plt.rcParams["figure.figsize"] = [15, 10]

    plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
    plot_data(X, y,Name=Name,s=s)

def plot_data(X, y, Name,zoom_out=False, s=None):
    if zoom_out:
        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])

        plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])

    plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=matplotlib.colors.ListedColormap(['blue', 'red']))
    plt.title(Name)
    plt.show()
def svm_dual_kernel(X, y, ker, max_iter=4000, verbose=False,d=0,g=0):
    N = X.shape[0]
    P = np.empty((N, N))
    for i, j in itertools.product(range(N), range(N)):
        if d!=0:
            P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :],degree=d)
        else:
            if g!=0:
                P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :],gamme=g)
            else:
                P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :])
    P = 0.5 * (P + P.T)
    P = 0.5 * P
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    return alpha

# Kernel's :
def RBF_kernel(x, y):
    return np.e ** (-(x - y).T @ (x - y))
def linear_kernel(X1, X2):
    return np.dot(X1, X2.T)
def polynomial_kernel(X1, X2, degree=2):
    return (1 + np.dot(X1, X2.T)) ** degree
def gaussian_kernel(X1, X2, gamme=1.0):
    return np.exp(-np.linalg.norm(X1-X2)**2 / 2 * (gamme**2))


if __name__ == "__main__":
    # Load the Data
    path = r"C:\Users\97252\Desktop\שיעורי בית\שנה ב\סמסטר ב\מבוא ללמידת מכונה\תרגיל 4\simple_nonlin_classification.csv"
    df = pd.read_csv(path)


    y = np.array(df['y'])
    X = np.array(df.drop(columns=['y']))

    # Split and shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    X_train=np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    loss_values=[]
    # Kernel names
    kernel_names = ['Polynomial Degree 2', 'Polynomial Degree 4', 'RBF', 'Gaussian (Gamma = 0.1)',
                    'Gaussian (Gamma = 1)', 'Gaussian (Gamma = 10)']


    # Linear kernel
    alpha = svm_dual_kernel(X_train, y_train,ker=linear_kernel)
    if alpha is not None and alpha.all():
        plot_classifier_z_kernel(alpha, X_train, y_train, ker=linear_kernel,Name='Linear kernal', s=80)
    else:
        print("Cant find the alpha in Linear kernal !")



    # Polynomial kernel
    #Degree 2
    alpha = svm_dual_kernel(X_train, y_train,ker=polynomial_kernel,d=2 )
    plot_classifier_z_kernel(alpha, X_train, y_train, ker=polynomial_kernel,d=2,Name='polynomial kernel degree = 2', s=80)
    predicts = predict(X_test=X_test, X_train=X_train, y_train=y_train, kernel=polynomial_kernel, alpha=alpha,d=2)
    loss = calculate_loss(y_true=y_test, y_pred=predicts)
    loss_values.append(loss)

    # Degree 4
    alpha = svm_dual_kernel(X_train, y_train, ker=polynomial_kernel, d=4)
    plot_classifier_z_kernel(alpha, X_train, y_train, ker=polynomial_kernel, d=4, Name='polynomial kernel degree = 4',
                             s=80)
    predicts = predict(X_test=X_test, X_train=X_train, y_train=y_train, kernel=polynomial_kernel, alpha=alpha, d=4)
    loss = calculate_loss(y_true=y_test, y_pred=predicts)
    loss_values.append(loss)



    # RBF kernel
    alpha = svm_dual_kernel(X_train, y_train, ker=RBF_kernel)
    plot_classifier_z_kernel(alpha, X_train, y_train, ker=RBF_kernel, Name='RBF_kernel',
                             s=80)
    predicts=predict(X_test=X_test,X_train=X_train,y_train=y_train,kernel=RBF_kernel,alpha=alpha)
    loss=calculate_loss(y_true=y_test,y_pred=predicts)
    loss_values.append(loss)

    # gaussian kernel
    # Gamma = 0.1
    alpha = svm_dual_kernel(X_train, y_train, ker=gaussian_kernel,g=0.1)
    plot_classifier_z_kernel(alpha, X_train, y_train, ker=gaussian_kernel, Name='gaussian kernel gamma = 0.1',
                             s=80,g=0.1)
    predicts = predict(X_test=X_test, X_train=X_train, y_train=y_train, kernel=gaussian_kernel, alpha=alpha,g=0.1)
    loss = calculate_loss(y_true=y_test, y_pred=predicts)
    loss_values.append(loss)

    # Gamma = 1
    alpha = svm_dual_kernel(X_train, y_train, ker=gaussian_kernel, g=1)
    plot_classifier_z_kernel(alpha, X_train, y_train, ker=gaussian_kernel, Name='gaussian kernel gamma = 1',
                             s=80, g=1)
    predicts = predict(X_test=X_test, X_train=X_train, y_train=y_train, kernel=gaussian_kernel, alpha=alpha, g=1)
    loss = calculate_loss(y_true=y_test, y_pred=predicts)
    loss_values.append(loss)


    # Gamma = 10

    alpha = svm_dual_kernel(X_train, y_train, ker=gaussian_kernel, g=10)
    plot_classifier_z_kernel(alpha, X_train, y_train, ker=gaussian_kernel, Name='gaussian kernel gamma = 10',
                             s=80, g=10)
    predicts = predict(X_test=X_test, X_train=X_train, y_train=y_train, kernel=gaussian_kernel, alpha=alpha, g=10)
    loss = calculate_loss(y_true=y_test, y_pred=predicts)
    loss_values.append(loss)

    # Plot the loss bar
    plot_loss_bar(loss_values,kernel_names)






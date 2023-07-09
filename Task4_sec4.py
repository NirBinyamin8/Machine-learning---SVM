from Task4_sec3 import SVM
from Task4_sec2 import linear_kernel
from Task4_sec2 import RBF_kernel
from Task4_sec2 import polynomial_kernel
import matplotlib.pyplot as plt
from Task4_sec2 import gaussian_kernel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def restart_model(model):
    model.kernel = linear_kernel
    model.degree = 0
    model.C = 0
    model.gamma = 0
    model.X_train = None
    model.y_train = None
    model.alpha = None
def plot_score_bar(loss_values,kernel_names):
    # Create a bar plot
    plt.bar(kernel_names, loss_values)
    plt.xlabel('Kernel')
    plt.ylabel('Score')
    plt.title('Score for Different Kernels')
    plt.xticks(rotation=20)
    plt.xticks(fontsize=8)


    # Display the plot
    plt.show()
if __name__ == "__main__":
    # Load the Data
    path = r"C:\Users\97252\Desktop\שיעורי בית\שנה ב\סמסטר ב\מבוא ללמידת מכונה\תרגיל 4\Processed Wisconsin Diagnostic Breast Cancer (1).csv"
    df = pd.read_csv(path)
    # Changing the representation of tags
    y = np.array(df['diagnosis'])
    y = y * 2 - 1
    X = np.array(df.drop(columns=['diagnosis']))
    # Split and shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    scores=[]
    kernel_names=[]

    # Create svm model
    model=SVM()

    # Linear Kernel :
    model.kernel=linear_kernel
    model.fit(X_train,y_train)
    if model.alpha is not None and model.alpha.all():
        scores.append(model.score(X_test,y_test))
        kernel_names.append('Linear kernel')
    else:
        print("Cant find the alpha in Linear kernal !")

    restart_model(model)


    # RBF Kernel :
    model.kernel=RBF_kernel
    model.fit(X_train, y_train)
    if model.alpha is not None and model.alpha.all():
        scores.append(model.score(X_test,y_test))
        kernel_names.append('RBF Kernel')
    else:
        print("Cant find the alpha in RBF Kernel !")

    restart_model(model)

    # Polynomial Kernel
    Degree=[i+1 for i in range(5)]
    for d in Degree:
        model.kernel = polynomial_kernel
        model.degree=d
        model.fit(X_train, y_train)
        if model.alpha is not None and model.alpha.all():
            scores.append(model.score(X_test, y_test))
            kernel_names.append('Polynomial Kernel Degree = '+str(d))
        else:
            print('Cant find the alpha in Polynomial Kernel Degree = '+str(d))
        restart_model(model)



    # Gaussian Kernel
    Gamma=[0.0001,0.001,0.01,0.1,1]
    for g in Gamma:
        model.kernel = gaussian_kernel
        model.gamma = g
        model.fit(X_train, y_train)
        if model.alpha is not None and model.alpha.all():
            scores.append(model.score(X_test, y_test))
            kernel_names.append('Gaussian Kernel Gamma = ' + str(g))
        else:
            print('Cant find the alpha in Gaussian Kernel Gamma = ' + str(g))
        restart_model(model)

    # Plot bar
    plot_score_bar(scores, kernel_names)

    # Best Kernel :
    print('The best kernel is :' +kernel_names[scores.index(max(scores))] +' With score = ',"{:.3f}".format(max(scores)))








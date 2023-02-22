import numpy as np
import matplotlib.pyplot as plt


def predict_derivative(regressor, X):
    # Predict derivatives using the learned model
    X_predict = regressor.predict(X)
    # Compute derivatives with a finite difference method
    X_dot_predict = regressor.model_.differentiate(X_predict, t=1)
    return X_dot_predict


def present_results(regressor, X):
    y_predict = regressor.predict(X)
    if len(X.shape) == 1:
        x = np.expand_dims(X, axis=1)
    if X.shape[1] == 1:
        plt.figure(figsize=(8, 4))
        plt.xlabel("Time")
        plt.ylabel("")
        plt.plot(regressor.t, X)
        plt.plot(regressor.t, y_predict)
    else:
        fig, axs = plt.subplots(X.shape[1], 1, figsize=(7, 9))
        for i in range(X.shape[1]):
            axs[i].plot(regressor.t_, x[:, i],
                        'k', label='model original')
            axs[i].plot(regressor.t_, y_predict[:, i],
                        'r--', label='model prediction')
            axs[i].legend()
            axs[i].set(xlabel='t', ylabel='$y_{}$'.format(i + 1))
    plt.show()

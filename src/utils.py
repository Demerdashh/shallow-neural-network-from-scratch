# we will use this function later to plot the decision boundary after finishing the model.
def plot_decision_boundary(model_func, X, Y):

    # Set min and max values and give some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01  # step size in the mesh

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict the function value for the whole grid
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # shape (2, number_of_points)
    Z = model_func(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot contour and training examples
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.7)
    plt.scatter(X[0, :], X[1, :], c=Y[0], cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

# Logistic Regression sigmoid function for the final output layer
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/ (1+np.exp(-z))

    return s

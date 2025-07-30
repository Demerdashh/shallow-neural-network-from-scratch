from sklearn.datasets import make_moons # only imported to create the Dataset

# The input and the labels
X, Y = make_moons(n_samples=400, noise=0.2)
X = X.T # transposes the X.shape to be (dim,m)
Y = Y.reshape(1, -1) #transposes the Y.shape to be (dim,m)

# Visualizing the data
plt.scatter(X[0, :], X[1, :], c=Y[0], cmap=plt.cm.Spectral)
plt.title("Moons Dataset")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#### The plot is in the output file

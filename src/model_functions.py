def layer_size(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """

    n_x = X.shape[0] # dim
    n_h = 5  # lets use 5 hidden layers
    n_y = Y.shape[0]

    return (n_x , n_h , n_y)

def initialize_parameters(n_x,n_h,n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of the first layer of shape (n_h, n_x)
                    b1 -- bias vector of the first layer of shape (n_h, 1)
                    W2 -- weight matrix of the second layer of shape (n_y, n_h)
                    b2 -- bias vector of  the second layer of shape (n_y, 1)
    """   

    W1 = np.random.randn(n_h,n_x) * 0.01  # we multiply by 0.01 because we will use the tanh() 
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)* 0.01
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Access each parameter from the dict "parameters"
    W1 = parameters["W1"]  #shape (n_h,n_x)
    b1 = parameters["b1"]  #shape (n_h,1)
    W2 = parameters["W2"]  #shape (n_y,n_h)
    b2 = parameters["b2"]  #shape (n_y,1)

    Z1 = (np.dot(W1,X)+b1)
    A1 = np.tanh(Z1)
    Z2 = (np.dot(W2,A1) + b2)
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    
    """

    m = Y.shape[1] # number of examples

    # we can use 2 approaches to do that 1 to use np.multiply() and np.sum() or to use np.dot() directly .
    # here I will use the np.multiply() and np.sum()

    loglos = (1/m)*(np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y))
    cost = - np.sum(loglos)

    # the other approach was that:
    ## cost = -(np.dot(Y.T, np.log(A2)) + np.dot((1 - Y).T, np.log(1 - A2))) / m

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. 
                                    # e.g., turns [[17]] into 17 
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1] # number of examples

    # Access each parameter from the dict "parameters" and "cache"
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache['A1']
    A2 = cache['A2']

     # Backward propagation: calculate dldW1, dldb1, dldW2, dldb2.

    dldZ2 = A2 - Y
    dldW2 = np.dot(dldZ2,A1.T)/m
    dldb2 = np.sum(dldZ2,axis =1,keepdims= True)/m
    dldZ1 = np.dot(W2.T,dldZ2)*(1 - np.power(A1, 2))
    dldW1 = (np.dot(dldZ1,X.T))/m
    dldb1 = np.sum(dldZ1,axis =1, keepdims = True)/m

    grads = {"dW1": dldW1,
             "db1": dldb1,
             "dW2": dldW2,
             "db2": dldb2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    W1 = copy.deepcopy(parameters["W1"]) # using deepcopy 
    b1 = parameters["b1"]
    W2 = copy.deepcopy(parameters["W2"]) # using deepcopy 
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"] 
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

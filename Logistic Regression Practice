import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

%matplotlib inline

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")

print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

# Excercise 1 apply sigmoid function

# UNQ_C1
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
          
    ### START CODE HERE ### 
    
    # Applying the sigmoid funcion
    g = 1/(1+np.exp(-z))
    
    ### END SOLUTION ###  
    
    return g
    
# Note: You can edit this value
value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")



print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)

# All tests passed




# Exercise 2 apply cost function for logistic regression

# UNQ_C2
# GRADED FUNCTION: compute_cos
t
def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ###
    
    # set loss_sum to zero
    loss_sum = 0
    
    # for loop for each training example
    for i in range(m): 
        z_wb = 0
        for j in range(n): 
            z_wb_ij = w[j]*X[i][j] 
            z_wb += z_wb_ij
        # Add the bias term to z_wb
        z_wb += b
        f_wb = sigmoid(z_wb) 
        loss =  -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        
        loss_sum += loss # equivalent to loss_sum = loss_sum + loss
        total_cost = (1 / m) * loss_sum  
        
    ### END CODE HERE ### 
    return total_cost
    
    m, n = X_train.shape

# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))

# Compute and display cost with non-zero w
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w,b: {:.3f}'.format(cost))


# UNIT TESTS
compute_cost_test(compute_cost)




# Excercise 3 
# Gradient for Logistic Regression

# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) values of parameters of the model      
      b : (scalar)                 value of parameter of the model 
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):
        z_wb = None
        for j in range(n): 
            z_wb += None
        z_wb += None
        f_wb = None
        
        dj_db_i = None
        dj_db += None
        
        for j in range(n):
            dj_dw[j] = None
            
    dj_dw = None
    dj_db = None
    ### END CODE HERE ###


    return dj_db, dj_dw


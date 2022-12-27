import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)

"""
Cost
- The cost functions differ significantly between linear and logistic regression, but adding regularization to the equations is the same.

Gradient
- The gradient functions for linear and logistic regression are very similar. They differ only in the implementation of 𝑓𝑤𝑏.
"""

#FIXME:
"""
Cost functions with regularization

Cost function for regularized linear regression
The equation for the cost function regularized linear regression is:
𝐽(𝐰,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))2+𝜆2𝑚∑𝑗=0𝑛−1𝑤2𝑗(1)
where:
𝑓𝐰,𝑏(𝐱(𝑖))=𝐰⋅𝐱(𝑖)+𝑏(2)
Compare this to the cost function without regularization (which you implemented in a previous lab), which is of the form:

𝐽(𝐰,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))2
The difference is the regularization term, 𝜆2𝑚∑𝑛−1𝑗=0𝑤2𝑗

Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter 𝑏 is not regularized. This is standard practice.

Below is an implementation of equations (1) and (2). Note that this uses a standard pattern for this course, a for loop over all m examples.
"""




def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
​
    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2                               #scalar             
    cost = cost / (2 * m)                                              #scalar  
 
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar


np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
​
print("Regularized cost:", cost_tmp)
Expected Output:

Regularized cost: 0.07917239320214275


#FIXME:
"""
Cost function for regularized logistic regression

For regularized logistic regression, the cost function is of the form
𝐽(𝐰,𝑏)=1𝑚∑𝑖=0𝑚−1[−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))]+𝜆2𝑚∑𝑗=0𝑛−1𝑤2𝑗(3)
where:
𝑓𝐰,𝑏(𝐱(𝑖))=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝐰⋅𝐱(𝑖)+𝑏)(4)
Compare this to the cost function without regularization (which you implemented in a previous lab):

𝐽(𝐰,𝑏)=1𝑚∑𝑖=0𝑚−1[(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))]
As was the case in linear regression above, the difference is the regularization term, which is 𝜆2𝑚∑𝑛−1𝑗=0𝑤2𝑗

Including this term encourages gradient descent to minimize the size of the parameters. Note, in this example, the parameter 𝑏 is not regularized. This is standard practice.
"""


def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """
​
    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar
​
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar


np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
​
print("Regularized cost:", cost_tmp)

# Expected Output:
# Regularized cost: 0.6850849138741673


#FIX ME:
"""
Gradient descent with regularization
The basic algorithm for running gradient descent does not change with regularization, it is:
repeat until convergence:{𝑤𝑗=𝑤𝑗−𝛼∂𝐽(𝐰,𝑏)∂𝑤𝑗𝑏=𝑏−𝛼∂𝐽(𝐰,𝑏)∂𝑏}for j := 0..n-1(1)
Where each iteration performs simultaneous updates on 𝑤𝑗 for all 𝑗.

What changes with regularization is computing the gradients.

Computing the Gradient with regularization (both linear/logistic)
The gradient calculation for both linear and logistic regression are nearly identical, differing only in computation of 𝑓𝐰𝑏.
∂𝐽(𝐰,𝑏)∂𝑤𝑗∂𝐽(𝐰,𝑏)∂𝑏=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))𝑥(𝑖)𝑗+𝜆𝑚𝑤𝑗=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))(2)(3)
m is the number of training examples in the data set
𝑓𝐰,𝑏(𝑥(𝑖)) is the model's prediction, while 𝑦(𝑖) is the target
For a linear regression model
𝑓𝐰,𝑏(𝑥)=𝐰⋅𝐱+𝑏
For a logistic regression model
𝑧=𝐰⋅𝐱+𝑏
𝑓𝐰,𝑏(𝑥)=𝑔(𝑧)
where 𝑔(𝑧) is the sigmoid function:
𝑔(𝑧)=11+𝑒−𝑧
The term which adds regularization is the 𝜆𝑚𝑤𝑗.
"""



# Gradient function for regularized linear regression

def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
​
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]                 
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]               
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m   
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
​
    return dj_db, dj_dw


np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
​
print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

# Expected Output:
# dj_db: 0.6648774569425726
# Regularized dj_dw:
#  [0.29653214748822276, 0.4911679625918033, 0.21645877535865857]
 
 
 
# Gradient function for regularized logistic regression

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar
​
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
​
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]
​
    return dj_db, dj_dw  


np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
​
print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

# Expected Output: 
# dj_db: 0.341798994972791
# Regularized dj_dw:
#  [0.17380012933994293, 0.32007507881566943, 0.10776313396851499]

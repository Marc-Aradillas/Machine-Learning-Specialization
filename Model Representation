# Model Representation lab

#Tools
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

# create your x_train and y_train variables. The data is stored in one-dimensional NumPy arrays.

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# use m to denote the number of training examples. Numpy arrays have a .shape parameter. x_tra
# in.shape returns a python tuple with an entry for each dimension. x_train.shape[0] is the length
# of the array and number of examples as shown below.

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

#One can also use the Python len() function as shown below.
# m is the number of training examples
		"""
		m = len(x_train)
		print(f"Number of training examples is: {m}")
		"""

# use (x(𝑖), y(𝑖)) to denote the 𝑖𝑡ℎ training example. Since Python is zero indexed, (x(0), y(0)) is (1.0, 300.0) and (x(1), y(1)) is (2.0, 500.0).
# To access a value in a Numpy array, one indexes the array with the desired offset. For example the syntax to access location zero of x_train is 
# x_train[0]. Run the next code block below to get the 𝑖𝑡ℎ training example.

i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# You can use other functions in the matplotlib library to set the title and labels to display

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='b')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
		
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")

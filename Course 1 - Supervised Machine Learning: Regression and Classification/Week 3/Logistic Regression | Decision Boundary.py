import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')

# DATASET
# Let's suppose you have following training dataset
# - The input variable X is a numpy array which has 6 training examples, each with two features
# - The output variable y is also a numpy array with 6 examples, and y is either 0 or 1

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

# PLOT DATA
# Let's use a helper function to plot this data. The data points with label 𝑦=1 are shown as
# red crosses, while the data points with label 𝑦=0 are shown as blue circles.

fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()

"""
LOGISTIC REGRESSION MODEL

- Suppose you'd like to train a logistic regression model on this data which has the form

    𝑓(𝑥) = 𝑔(𝑤0𝑥0 + 𝑤1𝑥1+𝑏)

    where 𝑔(𝑧) = 1 / 1 + 𝑒−𝑧, which is the sigmoid function

- Let's say that you trained the model and get the parameters as 𝑏 = −3, 𝑤0 = 1,𝑤1=1 That is,

    𝑓(𝑥) = 𝑔(𝑥0 + 𝑥1−3)

    (You'll learn how to fit these parameters to the data further in the course)

Let's try to understand what this trained model is predicting by plotting its decision boundary
"""

# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)

"""
- when plotted you can see,  𝑔(𝑧) >= 0.5  for  𝑧 >= 0 
- For a logistic regression model,  𝑧 = 𝐰⋅𝐱 + 𝑏 . Therefore,

    if  𝐰⋅𝐱+𝑏>=0 , the model predicts  𝑦 = 1 
    
    if  𝐰⋅𝐱+𝑏<0 , the model predicts  𝑦 = 0 
_____________________________________________________________
"""

"""
PLOTTING DECISION BOUNDARY

Let's go back to our example to understand how the logistic regression model is making predictions.

- Our logistic regression model has the form

    𝑓(𝐱) = 𝑔(−3 + 𝑥0 + 𝑥1)

- From what you've learnt above, you can see that this model predicts 𝑦 = 1 if −3 + 𝑥0 + 𝑥1 >= 0

Let's see what this looks like graphically. We'll start by plotting −3 + 𝑥0 + 𝑥1 = 0, which is equivalent to 𝑥1 = 3 − 𝑥0.
"""

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

"""
- In the plot above, the blue line represents the line 𝑥0 + 𝑥1 − 3 = 0 and it should intersect the x1 axis at 3
  (if we set 𝑥1 = 3, 𝑥0 = 0) and the x0 axis at 3 (if we set 𝑥1 = 0, 𝑥0 = 3).
  
- The shaded region represents −3 + 𝑥0 + 𝑥1 < 0. The region above the line is −3 + 𝑥0 + 𝑥1 > 0.

- Any point in the shaded region (under the line) is classified as 𝑦 = 0. Any point on or above the line is classified as 𝑦 = 1. This line is known as the
  "decision boundary".
  
As we've seen in the lectures, by using higher order polynomial terms (eg: 𝑓(𝑥) = 𝑔(𝑥20 + 𝑥1 − 1), we can come up with more complex non-linear boundaries.
"""

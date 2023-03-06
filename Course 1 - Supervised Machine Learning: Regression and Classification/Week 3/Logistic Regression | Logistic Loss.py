import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('./deeplearning.mplstyle')

"""
Recall for Linear Regression we have used the squared error cost function: The equation for the squared error cost with one variable is:

  𝐽(𝑤,𝑏)=1/2𝑚 m-1/∑ 𝑖=0 (𝑓𝑤,𝑏(𝑥(𝑖)) − 𝑦(𝑖))2 (1)

where

𝑓𝑤,𝑏(𝑥(𝑖)) = 𝑤𝑥(𝑖) + 𝑏  (2) 

Recall, the squared error cost had the nice property that following the derivative of the cost leads to the minimum.
"""

soup_bowl()

# This cost function worked well for linear regression, it is natural to consider it for logistic regression as well. However, as the slide above points out, 𝑓𝑤𝑏(𝑥)
# now has a non-linear component, the sigmoid function: 𝑓𝑤,𝑏(𝑥(𝑖))=𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑤𝑥(𝑖)+𝑏)fw,b(x(i))=sigmoid(wx(i)+b). Let's try a squared error cost on the example from an
# earlier lab, now including the sigmoid.

# Training Data

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train

"""
Now, let's get a surface plot of the cost using a squared error cost:

       𝐽(𝑤,𝑏)=1 / 2𝑚 m-1 ∑ 𝑖=0 (𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2
       
where

𝑓𝑤,𝑏(𝑥(𝑖)) = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑤𝑥(𝑖) + 𝑏)

plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()

# While this produces a pretty interesting plot, the surface above not nearly as smooth as the 'soup bowl' from linear regression!
# Logistic regression requires a cost function more suitable to its non-linear nature.

"""
LOGISTIC LOSS FUNCTION

Logistic Regression uses a loss function more suited to the task of categorization where the target is 0 or 1 rather than any number.

  |Definition Note: In this course, these definitions are used:
  |Loss is a measure of the difference of a single example to its target value while the
  |Cost is a measure of the losses over the training set

This is defined:

- 𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖)) is the cost for a single data point, which is:

                𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖)) = { −log (𝑓𝐰,𝑏(𝐱(𝑖)))   if 𝑦(𝑖) = 1
                                       { −log(1 − 𝑓𝐰,𝑏(𝐱(𝑖)))   if 𝑦(𝑖)=0
                
- 𝑓𝐰,𝑏(𝐱(𝑖)) is the model's prediction, while 𝑦(𝑖) is the target value.

- 𝑓𝐰,𝑏(𝐱(𝑖))=𝑔(𝐰⋅𝐱(𝑖)+𝑏) where function 𝑔 is the sigmoid function.

The defining feature of this loss function is the fact that it uses two separate curves. One for the case when the target is zero or (𝑦=0) and another for when
the target is one (𝑦=1). Combined, these curves provide the behavior useful for a loss function, namely, being zero when the prediction matches the target
and rapidly increasing in value as the prediction differs from the target. Consider the curves below:
"""

plt_two_logistic_loss_curves()

"""
Combined, the curves are similar to the quadratic curve of the squared error loss. Note, the x-axis is 𝑓𝐰,𝑏 which is the output of a sigmoid. The sigmoid output
is strictly between 0 and 1.

The loss function above can be rewritten to be easier to implement.

𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖)) = (−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖))) − (1−𝑦(𝑖))log(1 − 𝑓𝐰,𝑏(𝐱(𝑖)))

This is a rather formidable-looking equation. It is less daunting when you consider 𝑦(𝑖) can have only two values, 0 and 1. One can then consider the equation 
in two pieces:

when 𝑦(𝑖) = 0, the left-hand term is eliminated:           𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),0) = (−(0)log(𝑓𝐰,𝑏(𝐱(𝑖))) − (1 − 0)log(1 − 𝑓𝐰,𝑏(𝐱(𝑖))) 
                                                                                = −log(1 − 𝑓𝐰,𝑏(𝐱(𝑖)))

and when 𝑦(𝑖) = 1, the right-hand term is eliminated:      𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),1) = (−(1)log(𝑓𝐰,𝑏(𝐱(𝑖))) − (1 − 1)log(1 − 𝑓𝐰,𝑏(𝐱(𝑖)))
                                                                                = −log(𝑓𝐰,𝑏(𝐱(𝑖)))

OK, with this new logistic loss function, a cost function can be produced that incorporates the loss from all the examples. This will be the topic of the next lab.
For now, let's take a look at the cost vs parameters curve for the simple example we considered above:
"""

plt.close('all')
cst = plt_logistic_cost(x_train,y_train)

"""
This curve is well suited to gradient descent! It does not have plateaus, local minima, or discontinuities. Note, it is not a bowl as in the case of squared error.
Both the cost and the log of the cost are plotted to illuminate the fact that the curve, when the cost is small, has a slope and continues to decline.
"""

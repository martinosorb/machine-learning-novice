---
title: "Regression"
teaching: 45
exercises: 30
questions:
- "How can I make linear regression models from data?"
- "How can I use logarithmic regression to work with some kinds of non-linear data?"
- "How can I use multilinear regression?"
objectives:
- "Learn how to use linear regression to produce a model from data."
- "Learn how to model non-linear data using a logarithmic."
- "Learn how to model functions with multiple inputs."
- "Learn how to measure the error between the original data and a linear model." 
keypoints:
- "We can model linear data using a linear or least squares regression."
- "A linear regression model can be used to predict values within the range of validity."
- "We should split up our training dataset and use part of it to test the model."
- "For non-linear data we can use logarithms to make the data linear."
---

# Linear regression

If we take two variable and graph them against each other we can look for relationships between them. Once this relationship is established we can use that to produce a model which will help us predict values of one variable given the other. 

If the two variables form a linear relationship (a straight line can be drawn to link them) then we can create a linear equation to link them. This will be of the form y = m * x + c, where x is the variable we know, y is the variable we're calculating, m is the slope of the line linking them and c is the point at which the line crosses the y axis (where x = 0). 

[Kepler observed the following relationships between the mean distance of a planet from the Sun to it's period](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion)

| Planet  |	Mean distance to sun (AU) |	Period (days) |
|--       | --                        |--             |
| Mercury |	0.389                     |	87.77         |
| Venus 	| 0.724 	                  | 224.70        | 	
| Earth 	| 1 	                      | 365.25        |	
| Mars 	  | 1.524 	                  | 686.95        |
| Jupiter |	5.20 	                    | 4332.62       |
| Saturn 	| 9.510 	                  | 10759.2       |

Let us plot this data

~~~
import matplotlib.pyplot as plt


def make_plot(x_data, y_data, x_label, y_label):

    plt.scatter(x_data, y_data, label="Original Data")
    
    plt.grid()
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.save("planets_graph.svg")
    
x_data = [0.389,0.724,1,1.524,5.20,9.510]
y_data = [87.77,224.70,365.25,686.95,4332.62,10759.2]
x_label = "Mean distance to sun (AU)"
y_label = "Period (days)"
make_plot(x_data, y_data, x_label, y_label)

~~~
{: .python}

![graph of planet period against distance from the sun](../fig/planets_graph.svg)

The graph shows that as the distance from the sun increases, so does the time to complete a single orbit.
However, the relationship is not linear.  One can use logarithms to see if there is a power law 
relationship.

> ## Logarithms Introduction
> Logarithms are the inverse of an exponent (raising a number by a power). 
> logb(a) = c 
> b^c = a
> For example:
> 2^5 = 32
> log2(32) = 5
> If you need more help on logarithms see the [Khan Academy's page](https://www.khanacademy.org/math/algebra2/exponential-and-logarithmic-functions/introduction-to-logarithms/a/intro-to-logarithms)
> {: .callout}

Therefore, one can plot

~~~
import matplotlib.pyplot as plt
import numpy as np

def make_plot(x_data, y_data, x_label, y_label):

    plt.scatter(x_data, y_data, label="Original Data")
    
    plt.grid()
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.save('planets_log_graph.svg')
    
log_x_data = np.log(x_data)
log_y_data = np.log(y_data)
x_label = "log(Mean distance to sun (AU))"
y_label = "log(Period (days))"
make_plot(log_x_data, log_y_data, x_label, y_label)

~~~
{: .python}

![graph of log planet period against log distance from the sun](../fig/planets_log_graph.svg)

In this case, there is an approximate linear relationship. The coefficients can also be calculated

## Coding a linear regression with Python 
This code will calculate a least squares or linear regression for us.

~~~
def least_squares(data):
    x_sum = 0
    y_sum = 0
    x_sq_sum = 0
    xy_sum = 0

    # the list of data should have two equal length columns
    assert len(data[0]) == len(data[1])
    assert len(data) == 2

    n = len(data[0])
    # least squares regression calculation
    for i in range(0, n):
        x = data[0][i]
        y = data[1][i]
        x_sum = x_sum + x
        y_sum = y_sum + y
        x_sq_sum = x_sq_sum + (x**2)
        xy_sum = xy_sum + (x*y)

    m = ((n * xy_sum) - (x_sum * y_sum))
    m = m / ((n * x_sq_sum) - (x_sum ** 2))
    c = (y_sum - m * x_sum) / n

    print("Results of linear regression:")
    print("m=", m, "c=", c)

    return m, c
~~~
{: .python}

Lets test our code.

~~~
least_squares([log_x_data,log_y_data)])
~~~
{: .python}

We should get the following results:

~~~
Results of linear regression:
m= 1.5031353477782914 c= 5.897898811978949
~~~

This validates Kepler's observation that `(Mean distance
to sun )^3 	∝ (Period )^2` 

### Testing the accuracy of a linear regression model

We now have a simple linear model for some data. It would be useful to test how accurate that model is. We can do this by computing the y value for every x value used in our original data and comparing the model's y value with the original. We can turn this into a single overall error number by calculating the root mean square (RMS), this squares each comparison, takes the sum of all of them, divides this by the number of items and finally takes the square root of that value. By squaring and square rooting the values we prevent negative errors from cancelling out positive ones. The RMS gives us an overall error number which we can then use to measure our model's accuracy with. The following code calculates RMS in Python. 

~~~
import math
def measure_error(data1, data2):
    assert len(data1) == len(data2)
    err_total = 0
    for i in range(0, len(data1)):
        err_total = err_total + (data1[i] - data2[i]) ** 2

    err = math.sqrt(err_total / len(data1))
    return err
~~~
{: .python}


To calculate the RMS for the test data we just used we need to calculate the y coordinate for every x coordinate that we had in the original data. 

~~~
# get the m and c values from the least_squares function
m, c = least_squares([log_x_data,log_y_data])

# create an empty list for the model y data
fitted_data = np.exp(c) * np.pow(x_data,m)

# calculate the error
print(measure_error(y_data,fitted_data))
~~~
{: .python}

This will output an error of 3.813100984286817, which means that on average the difference between our model and the real values is 3.813100984286817. If the model perfectly matches the data then the value will be zero.




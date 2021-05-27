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
- "We should test models with data that has not been used to fit them."
- "Transformations of the data may allow for fitting of linear models to nonlinear relationships."
---

# Linear regression

If we take two variable and graph them against each other we can look for relationships between them. Once this relationship is established we can use that to produce a model which will help us predict values of one variable given the other. 

If the two variables form a linear relationship (a straight line can be drawn to link them) then we can create a linear equation to link them. This will be of the form y = m * x + c, where x is the variable we know, y is the variable we're calculating, m is the slope of the line linking them and c is the point at which the line crosses the y axis (where x = 0). 

[Kepler observed the following relationships between the mean distance of a planet from the Sun to it's period](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion)

| Planet  | Mean distance to sun (AU) |	Period (days) |
|--       | --                        |--             |
| Mercury | 0.389                     |	87.77         |
| Venus   | 0.724                     | 224.70        | 	
| Earth   | 1 	                      | 365.25        |	
| Mars 	  | 1.524                     | 686.95        |
| Jupiter | 5.20                      | 4332.62       |
| Saturn  | 9.510                     | 10759.2       |

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

This validates Kepler's observation that `(Mean distance to sun )^3 ∝ (Period )^2` 

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
fitted_data = np.exp(c) * np.power(x_data,m)

# calculate the error
print(measure_error(y_data,fitted_data))
~~~
{: .python}

This will output an error of 3.813100984286817, which means that on average the difference between our model and the real values is 3.813100984286817. If the model perfectly matches the data then the value will be zero.

> # Predicting the Period of Pluto
> 
> Pluto also orbits the sun. Can you use the fitted model to prdict the period of Pluto?
> Some information on Pluto can be found on
> [Wikipedia](https://en.wikipedia.org/wiki/Pluto) as well as on 
> [NASA's website](https://solarsystem.nasa.gov/planets/dwarf-planets/pluto/in-depth/)  
> > ## Solution
> > From the above links, Pluto has an average distance from the Sun 
> > of 39.5 AU.
> > 
> > m= 1.5031353477782914 c= 5.897898811978949
> > ~~~
> > print( np.exp(5.897898811978949) * np.power(39.5,1.5031353477782914))
> > ~~~
> > {: .python}
> > predicted period: 91480.05972244845 days, actual period 90520 days
> >
> > This is not bad, however, the input data is far outside the
> > range of the values used to fit the model.  Discuss whether
> > this could cause problems if used for prediction in other 
> > scenarios? What other reasons might account for this discrepancy?
> {: .solution}
{: .challenge}

> # An Updated Regression Model 
> 
> The [Wikipedia page on Kepler's laws of planetary motion](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion) 
> has updated measurements of the astronomical
> distance from the sun and the period which are listed below.
> | Planet  | Semi-major axis around the sun (AU) | Period (days) |
> |--       | --                                  |--             |
> | Mercury | 0.38710                             | 87.77         |
> | Venus   | 0.7233                              | 224.70        |
> | Earth   | 1                                   | 365.25        |
> | Mars    | 1.52366                             | 686.95        |
> | Jupiter | 5.20336                             | 4332.62       |
> | Saturn  | 9.510                               | 10759.2       |
> | Uranus  | 19.1913                             | 30687.153     |
> | Neptune | 30.0690                             | 60190.03      |
> Use this data to fit a regression model and estimate the period 
> of Pluto. 
> > ## Solution
> > 
> > ~~~
> > x_distance = [0.38710,0.7233,1,1.52366,5.20336,9.510,19.1913,30.0690]
> > y_period = [87.77,224.70,365.25,686.95,4332.62,10759.2,30687.153,60190.03]
> > m, c = least_squares([np.log(x_distance),np.log(y_period)])
> > pluto_au = 39.5
> > print( np.exp(c) * np.power(pluto_au,m))
> > ~~~
> > {: .python}
> >
> > Results of linear regression:
> > m= 1.5003670408045884 c= 5.900201729955274
> > predicted period: 90762.55504601273 days, actual period 90520 days
> > 
> > The results of this prediction are better than the previous one.
> > The method may still be unsatisfactory. Rudy, Brunton, Proctor  
> > and Kutz propose to use 
> > [Data-driven discovery of partial differential equations](https://arxiv.org/abs/1609.06401) 
> > as one method of fitting dynamical equations.
> >
> > The current fitted model does not account for the influence of the 
> > orbiting bodies on each other. Furthermore, 
> > [general relativity](https://en.wikipedia.org/wiki/Two-body_problem_in_general_relativity)
> > can better describe the orbits of bodies around the sun. Discuss 
> > how you might examine the data to separate effects of measurement 
> > errors from modelling errors in the choice of model to apply 
> > regression fitting to.
> {: .solution}
{: .challenge}

# Multilinear Regression

One can extend regression models to have more than one dependent variable. Rather, than
give full details of the derivation, a demonstration will be provided. Many datasets
contain correlated variables.  Regression can help determine correlation between variables.
A dataset of interest is the simulated 
[Major Atmospheric Gamma Imaging Cherenkov Telescope project (MAGIC)](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope) 
dataset. The dataset contains simulations which have data that distinguishes between 
detecting background noise and high energy gamma particles. Here, multilinear regression will 
be used to show that some of the measurements are correlated.

First setup the environment and read in the data and examine the first few entries
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
magic_data=pd.read_csv('../data/magic04.data',sep=',',header=None)
print(magic_data.head())
```

Then extract the row entries which correspond to gamma particles and drop
the column with the label for gamma particles
```python
magic_data_gamma = magic_data.query( '@magic_data[10] == "g"')
magic_data_gamma = magic_data_gamma.drop(
    magic_data_gamma.columns[[10]], axis = 1)
```

The resulting dataframe has the following columns:

    1.  fLength:  continuous  # major axis of ellipse [mm]
    2.  fWidth:   continuous  # minor axis of ellipse [mm] 
    3.  fSize:    continuous  # 10-log of sum of content of all pixels [in #phot]
    4.  fConc:    continuous  # ratio of sum of two highest pixels over fSize  [ratio]
    5.  fConc1:   continuous  # ratio of highest pixel over fSize  [ratio]
    6.  fAsym:    continuous  # distance from highest pixel to center, projected onto major axis [mm]
    7.  fM3Long:  continuous  # 3rd root of third moment along major axis  [mm] 
    8.  fM3Trans: continuous  # 3rd root of third moment along minor axis  [mm]
    9.  fAlpha:   continuous  # angle of major axis with vector to origin [deg]
   10.  fDist:    continuous  # distance from origin to center of ellipse [mm]

We will fit a regression model for fDist (the distance from origin to center of ellipse)
as a function of the other quantities. Therefore, split the dataframe into a single
column for fDist and a dataframe for the dependent variables, then use the scikit-learn
regression fitting routine and calculate the quality of the fit using the R^2 value.

```python
fDist = magic_data_gamma[[9]]
Variables = magic_data_gamma.drop(magic_data_gamma.columns[[9]], axis = 1)
reg = LinearRegression().fit(Variables,fDist)
print(reg.score(Variables,fDist))
```

The result of this gives an R^2 value of 0.42985458535037824 indicating some amount
of correlation between fDist and the other dependent variables.

> # Discussion
> 
> Multilinear regression has been applied without any modelling assumptions
> to fit a physics based model or derive a physics based model. What other
> work can you find on the internet that uses the 
> [Major Atmospheric Gamma Imaging Cherenkov Telescope project (MAGIC)](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)
> data?
> > ## Partial Solution
> > The original paper describing the collection of the data and its
> > analysis is
> > [Methods for multidimensional event classification: a case study using images from a Cherenkov gamma-ray telescope](http://www.cs.cas.cz/~savicky/papers/magic_case_study.pdf) 
> >
> > The dataset is also listed in the 
> > [Open Machine Learning project](https://www.openml.org/d/1120)
> > 
> {: .solution}
{: .challenge}

# Further Reading

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html


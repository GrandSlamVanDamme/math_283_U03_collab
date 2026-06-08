#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
data_fitting.py

This file contains functions for data fitting in MATH 283
author: Corey Silver
date: 3/6/26 (Yuro Style)

functions require numpy, scipy, matplotlib, sympy libraries
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import sympy as sym


def data_fit(X, Y):
    """
    The big Huncho Grande Paparoni: fitting, error analysis, plotting.
    """
    fit_type = input(
        "Please enter the fit type. Is it linear, polynomial, or exponential?"
    )

    if fit_type == "polynomial":
        n = int(
            input("What degree of polynomial? Please give your answer as a numeral.")
        )
    if fit_type == "exponential":
        X = np.log(X)
        Y = np.log(Y)
        n = 1
    else:
        n = 1

    plotter(X, Y, n)
    error_analyzer(fit_type)


def error_analyzer(model_name):
    """
    prints results of least squares, chebyshev, and absolute deviation analyses for
    type of model given by model_name
    """
    print(f"For the {model_name} model")
    print(f"We found a least squares error of {leastsquerror(Y, F)[-1]}")
    print(f"We found an absolute deviation error of {absdev_error(Y, F)[-1]}")
    print(f"We found a Chebyshev error of {cheb_error(Y, F)}")


def leastsquerror(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values].
    Computes cumulative variances of a series of residuals
    returns list of variances
    """
    resid = residuals(Y, F)

    squares = []

    squm = []

    for r in resid:
        squares.append(r**2)

        squm.append(sum(squares))

    return squm


def absdev_error(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values].
    computes cumulative absolute deviation
    returns list of absolute deviations
    """

    resid = residuals(Y, F)

    normals = []

    absum = []

    for r in resid:
        normals.append(abs(r))

        absum.append(sum(normals))

    return absum


def cheb_error(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values].
    returns maximum residual
    """

    cheb = abs(max(residuals(Y, F)))

    return cheb


def residuals(Y, F):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values]
    computes their residuals
    returns list of residuals

    """

    resid = []

    for f, y in zip(F, Y):
        resid.append(y - f)

    return resid


def arrayer(x, Y):
    """
    requires numpy
    takes two lists and makes them into arrays
    """

    x_array = np.array(x)
    Y_array = np.array(Y)

    return x_array, Y_array


def x_lists(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    returns objective function f
    """

    mat_x = []
    mat_y = []

    objective = []

    for Xi, Yi in zip(X, Y):
        rightrow = []
        leftrow = []
        bounds = []
        x_list = []

        while n > -1:
            rightrow.append(Xi**n)
            leftrow.append(-(Xi**n))
            x_list.append(Xi**n)

            bounds.append((None, None))
            objective.append(0)

            n -= 1

        rightrow.append(-1)
        leftrow.append(-1)
        mat_x.append(rightrow)
        mat_x.append(leftrow)
        mat_y.append(Yi)
        mat_y.append(-Yi)

    bounds[-1] = (0, None)
    objective.append(1)

    barry = [objective, mat_x, mat_y, bounds, x_list]

    return barry


def chebyshevify(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    Using Chebyshev criteria, fits model function
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    such that (max|Yi-Fi|, i ϵ NN) is minimized.
    returns F(x) as an array of coefficients, the last being the max error E

    ###########################################################################################################
    ##  https://byui.instructure.com/courses/409534/pages/w07-tuesday-lesson-plans-2?module_item_id=4500264  ##
    ###########################################################################################################

    used as guideline and partial template
    """

    """
    mat_x = []
    mat_y = []
    
    objective = []
    
    x = sym.symbol("x")
    
    for xi, Yi in zip(X, Y):
        
        rightrow = []
        leftrow = []
        bounds = []
        x_list = []
        
        while n > -1:
            
            rightrow.append(xi**n)
            leftrow.append(-(xi**n))
            x_list.append(x**n)
            
            bounds.append((None, None))
            objective.append(0)
            
            n -= 1
        
        rightrow.append(-1)
        leftrow.append(-1)
        mat_x.append(rightrow)
        mat_x.append(leftrow)
        mat_y.append(Yi)
        mat_y.append(-Yi)
    
    bounds[-1] = (0, None)
    objective.append(1)
    """

    barry = x_lists(X, Y, n)

    result = sci.linprog(
        barry[0], A_ub=barry[1], b_ub=barry[2], bounds=barry[3], method="highs"
    )

    cheb_fit = np.dot(barry[-1], (result.X).pop())

    return cheb_fit


def absdev_fit(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    Using fits model function
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    such that (sum|Yi-Fi|, i ϵ NN) is minimized.
    returns F(x) as an array
    """

    barry = x_lists(X, Y, n)

    A = sum(barry[1])
    b = sum(barry[2])

    result = sci.linprog(
        barry[0], A_ub=barry[1], b_ub=barry[2], bounds=barry[3], method="highs"
    )

    abs_dev_fit = np.dot(barry[-1], (result.X).pop())

    return abs_dev_fit


def plotter(X, Y, n):
    """
    Plots different fit functions for a given data fit type (linear, poly, etc)
    """
    LS2 = np.polyfit(X, Y, n)
    cheb = chebyshevify(X, Y, n)
    abs_dev = absdev_fit(X, Y, n)

    plt.scatter(X, Y)
    plt.plot(X, LS2)
    plt.plot(X, cheb)
    plt.plot(X, abs_dev)
    plt.legend()


if __name__ == "__data_fit__":
    main()
# In[ ]:

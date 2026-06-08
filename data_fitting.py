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


def main(x, Y, n):
    """
    The big Huncho Grande Paparoni: fitting, error analysis, plotting.
    """

    chebyshevify(x, Y, n)


def error_analyzer(Y, F):
    """
    Takes three lists: x = [ind. data points] Y = [dep. data points], and F(x) = [func_values].
    prints results of least squares, chebyshev, and absolute deviation analyses
    """
    print(f"For the model")
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

    return objective, mat_x, mast_y, bounds, x_list


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

    result = linprog(objective, A_ub=mat_x, b_ub=mat_y, bounds=bounds, method="highs")

    cheb_fit = np.dot(barry[-1], result.x)

    return cheb_fit


def absdev_fit(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    Using fits model function
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    such that (sum|Yi-Fi|, i ϵ NN) is minimized.
    returns F(x) as an array
    """

    absdev = x_lists(X, Y, n)


def conspiracy(x, Y, F):
    """
    Plots fit function
    """

    sym.plot(x, Y)
    sym.plot(F, ())


# In[ ]:

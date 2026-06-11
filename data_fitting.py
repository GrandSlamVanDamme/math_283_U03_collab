#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
data_fitting.py

This file contains functions for data fitting in MATH 283
author: Corey Silver
date: 3/6/26 (Yuro Style)

functions require numpy, scipy, matplotlib, sympy libraries

debugging suggestions (frequently wrong) from gemini: https://gemini.google.com/share/de2be7b1f7ee
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci


def main(X, Y, fit_type):
    """
    The big Huncho Grande Paparoni: fitting, error analysis, plotting.
    """
    """
    fit_type = input(
        "Please enter the fit type. Is it linear, polynomial, or exponential?"
    )
    """
    # fit_type = "linear"
    if fit_type == "polynomial":
        n = int(
            input("What degree of polynomial? Please give your answer as a numeral.")
        )
    if fit_type == "exponential":
        Y = np.log(Y)
        n = 1
    if fit_type == "power":
        Y = np.log(Y)
        X = np.log(X)
        n = 1
    if fit_type == "linear":
        n = 1
    else:
        print("Request cannot be completed, defaulting to rat")
        import matplotlib.image as mpimg

        img = mpimg.imread("fat_rats/Joanna_Servaes_wikimedia_commons.webp")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        return

    plotter(X, Y, n)
    error_analyzer(fit_type, X, Y, n)


def error_analyzer(model_name, X, Y, n, k=-1):
    """
    prints results of least squares, chebyshev, and absolute deviation analyses for
    type of model given by model_name and degree given by n. For error up to an arbitrary point, edit indices
    of the individual error functions. Default value of k is -1.
    """
    print(f"For the {model_name} model")
    print(
        f"We found a least squares error of {leastsquerror(Y, np.polyval(LS2_fit(X, Y, n), X))[k]}"
    )
    print(
        f"We found an absolute deviation error of {absdev_error(Y, np.polyval(absdev_fit(X, Y, n), X))[k]}"
    )
    print(
        f"We found a Chebyshev error of {cheb_error(Y, np.polyval(chebyshevify(X, Y, n), X))}"
    )


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


def objectivist(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    returns objective function f
    """

    mat_x = []
    mat_y = []
    x_list = []

    for Xi, Yi in zip(X, Y):
        i = n

        rightrow = []
        leftrow = []
        objective = []
        bounds = []

        while i > -1:
            bounds.append((None, None))
            objective.append(0)
            rightrow.append(Xi**i)
            leftrow.append(-(Xi**i))
            x_list.append(Xi**i)

            i -= 1

        rightrow.append(-1)
        leftrow.append(-1)
        mat_x.append(rightrow)
        mat_x.append(leftrow)
        mat_y.append(Yi)
        mat_y.append(-Yi)

    bounds.append((0, None))
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

    barry = objectivist(X, Y, n)

    result = sci.optimize.linprog(
        barry[0], A_ub=barry[1], b_ub=barry[2], bounds=barry[3], method="highs"
    )

    result = np.delete(result.x, -1)

    return result


def sum_maker(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2]
    and desired polynomial degree n. Returns a matrix of
    sums of x values and a matrix of sums of y values
    """

    i = n

    sum_list_you_got_there = []
    neg_sum_list = []

    while i > -1:
        power_list = []
        neg_pow_list = []

        for Xi in X:
            it = Xi**i
            power_list.append(it)
            neg_pow_list.append(-it)

        sum_list_you_got_there.append(sum(power_list))
        neg_sum_list.append(sum(neg_pow_list))

        i -= 1

    x_mat = [sum_list_you_got_there, neg_sum_list]
    y_mat = [sum(Y), -sum(Y)]

    return [x_mat, y_mat]


def absdev_fit(X, Y, n):
    """
    takes X data and makes a list of powers
    (X^n, X^n-1...),
    dots it with undefined params to whip up
    a cost function that absdev_fit
    will minimize

    #####################################################################################################################
    ##  https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model  ##
    #####################################################################################################################

    used as guideline and partial template
    """

    def cost_func(params):
        # sympy removal courtesy of gemini thread listed up top
        # SciPy will plug the current numerical guess into 'params'
        f = np.polyval(params, X)
        # Return the L1 norm (sum of absolute deviations)
        return np.sum(np.abs(Y - f))

    """
    f = np.polyval(params, X)
    cost_func = np.sum(np.abs(Y - f))
    """

    guess = (n + 1) * [1.0]

    result = sci.optimize.minimize(cost_func, guess, method="Nelder-Mead")

    return result.x


def LS2_fit(X, Y, n):
    """
    Takes two arrays, x = [data 1] and Y = [data 2].
    fits model function
    f(x) = c_n*x^n + c_(n-1)x^(n-1)+...c_0x^0 with n parameters
    such that (sum|Yi-Fi|^2, i ϵ NN) is minimized.
    returns f(x) as an array
    """
    LS2 = np.polyfit(X, Y, n)

    return LS2


def x_list(X, n):
    """
    takes X data and a polynomial degree n, returns linspaced list of X-powers
    from X^n to X^0.
    """
    X = np.linspace(X[0], X[-1] + np.mean(X) / 1000, 1000)

    return X


def plotter(X, Y, n):
    """
    Plots different fit functions for a given data fit type (linear, poly, etc)
    """
    LS2_coeffs = LS2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    abs_dev_coeffs = absdev_fit(X, Y, n)

    exes = x_list(X, n)

    print(f"Least squares fitting coeffs: {LS2_coeffs}")
    print(f"Chebyshev fit coeffs: {cheb_coeffs}")
    print(f"Absolute deviation fit coeffs: {abs_dev_coeffs}")

    LS2 = np.polyval(LS2_coeffs, exes)
    cheb = np.polyval(cheb_coeffs, exes)
    absdev = np.polyval(abs_dev_coeffs, exes)

    # Labels so I don't have to type them thrice

    xlab = "Hours passed from Reactor Shutdown"
    ylab = "Reactor temperature in $^{\\circ} C$"

    fig = plt.figure(figsize=(20, 40))

    plt.rcParams["font.size"] = 32
    # plt.rcParams["axes.labelpad"] = 4

    ax1 = plt.subplot(3, 1, 1)
    ax1.scatter(X, Y)
    ax1.plot(exes, LS2, color="r")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab, rotation=0, labelpad=(225))
    ax1.set_title("Least-Squares Fit")

    ax2 = plt.subplot(3, 1, 2)
    ax2.scatter(X, Y)
    ax2.plot(exes, cheb, color="y")
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab, rotation=0, labelpad=(225))
    ax2.set_title("Chebyshev Fit")

    ax3 = plt.subplot(3, 1, 3)
    plt.scatter(X, Y)
    plt.plot(exes, absdev, color="b")
    ax3.set_xlabel(xlab)
    ax3.set_ylabel(ylab, rotation=0, labelpad=(225))
    ax3.set_title("Absolute Deviation Fit")


# %%

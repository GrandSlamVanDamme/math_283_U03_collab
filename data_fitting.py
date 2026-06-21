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
import pandas as pd
import IPython.display as ipd
# import shiny as sh
# import shinywidgets as shw

func_types = ["linear", "polynomial", "power", "exponential"]
# poly_deg = 2
"""
poly_deg = int(
    input("What degree of polynomial? Please give your answer as a numeral.")
)
"""
degree_list = ["zeroth", "linear", "quadratic", "cubic", "quartic", "quintic"]
k = 0
while k < 95:
    degree_list.append(f"{k + 6}th degree")
    k += 1


def main(X, Y, n=2):
    """
    The big Huncho Grande Paparoni: fitting, error analysis, plotting.
    """
    # func_types = ["linear", "polynomial", "power", "exponential"]

    table_list = []

    for func in func_types:
        f = functionator(X, Y, func, n)
        X, Y, k, func_type = f[0:5]

        # plotter(X, Y, n, func_type)
        entry = error_analyzer(X, Y, k)
        table_list.append(entry)

    table_du_fromage = pd.DataFrame(
        table_list,
        index=pd.MultiIndex.from_product(
            [["Fit Type"], ["linear", degree_list[n], "power", "exponential"]]
        ),
        columns=pd.MultiIndex.from_product(
            [
                ["Optimization Type"],
                ["Least Squares", "Absolute Deviation", "Chebyshev"],
            ]
        ),
    )

    # ipd.display(table_du_fromage)

    # ipd.Markdown(table_du_fromage.to_markdown(index=False))
    return table_du_fromage


def functionator(X, Y, func_type, n):
    """
    Takes X, Y, and func type.
    Modifies X and Y if needed
    returns X, Y, n, func_type as list
    """
    fit_type = func_type
    k = n

    if fit_type == "polynomial":
        fit_type = degree_list[n]
    elif fit_type == "exponential":
        Y = np.log(Y)
        k = 1
    elif fit_type == "power":
        Y = np.log(Y)
        X = np.log(X)
        k = 1
    elif fit_type == "linear":
        k = 1
    else:
        print("Request cannot be completed, defaulting to rat")
        import matplotlib.image as mpimg

        img = mpimg.imread("fat_rats/Joanna_Servaes_wikimedia_commons.webp")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.axis("off")
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        fit_type = None
        return

    return [X, Y, k, fit_type]


def error_analyzer(X, Y, n, k=-1):
    """
    prints results of least squares, chebyshev, and absolute deviation analyses for
    type of model given by model_name and degree given by n. For error up to an arbitrary point, edit indices
    of the individual error functions. Default value of k is -1.
    """

    LS2 = leastsquerror(Y, np.polyval(LS2_fit(X, Y, n), X))[k]
    AbsDev = absdev_error(Y, np.polyval(absdev_fit(X, Y, n), X))[k]
    Cheby = cheb_error(Y, np.polyval(chebyshevify(X, Y, n), X))

    """
    print(f"For the {model_name} model")
    print(
        f"We found a least squares error of {LS2}"
    )
    print(
        f"We found an absolute deviation error of {AbsDev}"
    )
    print(
        f"We found a Chebyshev error of {Cheby}"
    )
    """

    return [LS2, AbsDev, Cheby]


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


def residuals(Y, F, func_type="linear"):
    """
    Takes two lists, Y = [data points] and F(x) = [func_values]
    computes their residuals
    returns list of residuals

    """

    resid = []
    if func_type == "exponential" or func_type == "power":
        Y = [np.e**y for y in Y]
    else:
        pass

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


def x_list(X):
    """
    takes X data, returns linspaced list of X-powers
    from X^n to X^0.
    """
    X = np.linspace(X[0], X[-1] + np.mean(X) / 1000, 1000)

    return X


def fitter_happier_better(X, Y, n):
    """
    Takes X, Y, and polynomial degree n.
    Grabs coefficients from the fit generators
    and returns arrays of actual function output.
    """
    LS2_coeffs = LS2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    abs_dev_coeffs = absdev_fit(X, Y, n)

    exes = x_list(X)

    # print(f"Least squares fitting coeffs: {LS2_coeffs}")
    # print(f"Chebyshev fit coeffs: {cheb_coeffs}")
    # print(f"Absolute deviation fit coeffs: {abs_dev_coeffs}")

    LS2 = np.polyval(LS2_coeffs, exes)
    cheb = np.polyval(cheb_coeffs, exes)
    absdev = np.polyval(abs_dev_coeffs, exes)

    return [exes, LS2, cheb, absdev]


def coeff_table(X, Y, func_type, n):
    """
    Takes X, Y, polynomial degree n.
    Grabs fits and returns their parameters
    for different func_types in a table
    format.
    """

    # for func in func_types:
    table_list = []

    f = functionator(X, Y, func_type, n)
    X, Y, n, func_type = f[0:5]

    LS2_coeffs = LS2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    abs_dev_coeffs = absdev_fit(X, Y, n)

    coeffs_list = [LS2_coeffs, cheb_coeffs, abs_dev_coeffs]

    c_list = [f"c{c}" for c in range(n + 1)]

    for coeff in coeffs_list:
        entry = coeff
        table_list.append(entry)

    table_du_fromage = pd.DataFrame(
        table_list,
        index=pd.MultiIndex.from_product(
            [
                ["Optimization Type"],
                ["Least Squares", "Absolute Deviation", "Chebyshev"],
            ]
        ),
        columns=pd.MultiIndex.from_product([["Function Parameters"], c_list]),
    )

    # ipd.display(table_du_fromage, display_id=func_type)

    # ipd.Markdown(table_du_fromage.to_markdown(index=False))

    return table_du_fromage


def plotter(X, Y, func_type, n, font_size=12):
    """
    Plots different fit functions for a given data fit type (linear, poly, etc)
    """
    X, Y, n, func_type = functionator(X, Y, func_type, n)[0:4]

    exes = fitter_happier_better(X, Y, n)[0]
    LS2 = fitter_happier_better(X, Y, n)[1]
    cheb = fitter_happier_better(X, Y, n)[2]
    absdev = fitter_happier_better(X, Y, n)[3]

    # Labels so I don't have to type them thrice

    xlab = "Hours passed since Reactor Shutdown"
    ylab = "Reactor temperature in $^{\\circ} C$"

    fig1 = plt.figure(figsize=(7, 20), layout="tight")
    # fig1.suptitle(f"Below are the optimizations for a {func_type} fit")

    plt.rcParams["font.size"] = font_size
    # plt.rcParams["axes.labelpad"] = 10

    dist = 40
    pointsize = 40
    point_color = "black"

    ax1 = plt.subplot(3, 1, 1)
    ax1.scatter(X, Y, s=pointsize, c=point_color)
    ax1.plot(exes, LS2, color="r")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab, rotation=55, labelpad=(dist), fontsize=12)
    ax1.set_title("Least-Squares Fit")

    ax2 = plt.subplot(3, 1, 2)
    ax2.scatter(X, Y, s=pointsize, c=point_color)
    ax2.plot(exes, cheb, color="y")
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab, rotation=55, labelpad=(dist), fontsize=12)
    ax2.set_title("Chebyshev Fit")

    ax3 = plt.subplot(3, 1, 3)
    plt.scatter(X, Y, s=pointsize, c=point_color)
    plt.plot(exes, absdev, color="b")
    ax3.set_xlabel(xlab)
    ax3.set_ylabel(ylab, rotation=55, labelpad=(dist), fontsize=12)
    ax3.set_title("Absolute Deviation Fit")


def temp_finder(X, Y, n):
    """
    Takes X, Y, n then obtains
    user input for time since
    reactor shutdown (Will add this feature later).
    Returns projected temp
    as calculated from Chebyshev-optimized
    exponential fit
    """

    """
    x = float(input("How many hours has it been since the reactor shut down"))
    """

    x = 24
    X, Y, n = functionator(X, Y, "exponential", n)[0:3]

    # LS2_coeffs = LS2_fit(X, Y, n)
    cheb_coeffs = chebyshevify(X, Y, n)
    # abs_dev_coeffs = absdev_fit(X, Y, n)

    # LS2 = np.polyval(LS2_coeffs, x)
    cheb = np.polyval(cheb_coeffs, x)
    # absdev = np.polyval(abs_dev_coeffs, x)

    T = f"After {x} hours, we project an estimated reactor temperature of {cheb:.3f} degrees centigrade."

    return T


X = [1, 2, 4, 8, 12, 18]
Y = [580, 510, 430, 340, 290, 230]

if __name__ == "__main__":
    main(X, Y)
coeff_table(X, Y, "linear", 1)
# temp_finder(X, Y)
# plotter(X, Y, "polynomial")
# %%

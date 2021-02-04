import math
from loewner_matrix import rational_interpolate
from polynomials_and_series import p_over_q_vals, series
from decimal import *

def rational_interpolate_function_vals(x_data, p, q, return_approximation = False):
    '''
    Let x_data = [x_1, x_2, ... , x_n]. Let p, q be some functions. 
    This function analyzes a sequence of values
         ___                                     ___
        |     p(x_0)     p(x_1)          p(x_n)    |
        |    ------     ------          ------     |
        |     q(x_0) ,   q(x_1)  , ... , q(x_n)    |
         ___                                     ___
    and finds a rational function 
                          p'(x)         
                f(x) = ----------
                          q'(x)
    which interpolates the values at the points x_0, x_1, .... , x_n.

    Since this is motivated with the goal of interpolating values 
    so that we can build a rational series, we also compute the infinite sum 
       \infty
        ----,
        \               p'(k)
        /     (1/10)^k ------ 
        ----`           q'(k)
        k = 0
    Our goal is to find p', q' so that the above sum equals pi. 

    ___________parameters___________
    x_data : (list of floats) x-coordinates for values of interest 
    numer  : (string) numerator of function values we wish to interpolate 
    denom  : (string) denominator of function values we wish to interpolate 

    Note that numer and denom need to be math functions that Python can understand. For example, 
    if a factorial appears in the expression, one  must write "math.factorial(x)" instead of "x!".
    '''
    # Evaluate function at points, collect them, call it y_data
    y_data = p_over_q_vals(p, q, x_data)
    
    # Interpolate the (x,y) data
    inter_p, inter_q, inter_p_exp, inter_q_exp = rational_interpolate(x_data, y_data)
    
    # Inform the reader of the polynomials obtained by interpolation (To do: make it more readable)
    # print("NUMERATOR \n", inter_p_exp, "\n")
    # print("DENOMINATOR \n", inter_q_exp, "\n")
    
    # Compute and display the infinite sum
    summands_for_series = p_over_q_vals(inter_p, inter_q, list(range(0, 100)), coeff = 1/16)
    print("SUM: ", Decimal(series(summands_for_series)))
    print("ERROR: ", Decimal(math.pi) - Decimal(series(summands_for_series)))
    if return_approximation:
        return inter_p_exp, inter_q_exp

p = "(16^x)"
q = "(4*x + 1)*(4*x + 3)"
import numpy as np
import math
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from numpy.polynomial import polynomial

def split_by_alternating(x_data):
    '''
    Takes a list [x_1, x_2, ...,  x_n] and returns two lists:
        [x_1, x_3, ..., x_{n-1}],[x_2, x_4, ..., x_{n}] (if n is even)
                                or    
        [x_1, x_3, ..., x_{n}],[x_2, x_4, ..., x_{n-1}] (if n is odd)
    ___________parameters___________
    x_data : list
    '''
    left = []
    right = []
    for i, pt in enumerate(x_data):
        if i % 2 == 0:
            left.append(pt)
        else:
            right.append(pt)
    return left, right

def split_in_half(x_data, index = None):
    '''
    Takes a list [x_1, x_2, ..., x_n] and returns two lists:
        [x_1, x_2, ..., x_{n/2}],[x_{n/2 + 1}, ..., x_{n}] (if n is even)
                                or    
        [x_1, x_3, ..., x_{(n+1)/2}],[x_{(n+1)/2 + 1}, ..., x_{n}] (if n is odd)
    ___________parameters___________
    x_data : list 
    index : integer.
            *Default is either n/2 or (n+1)/2 where n = len(x_data).
    '''
    if index is None:
        if len(x_data) % 2 == 0:
            index = int(len(x_data)/2)
        else:
            index = int((len(x_data) + 1)/2)
    left = x_data[0: index] # slice the list at the index
    right = x_data[index:]
    return left, right

def multiply_but_one(factors, index):
    '''
    Takes in a list of strings of the form 
        [ "(x - a_1)", "(x - a_2)", ..., "(x - a_n)"]
    and given i in {1, 2, ..., n}, multiply_but_one returns the string 
        _____
        |   |
        |   |  (x - a_j)
        j != i  
    Ex: factors = ["(x - 1)", "(x - 2)",  "(x - 3)", "(x - 4)"] 
        index   =  1
    would return  the string
        "(x - 1)(x - 3)(x - 4)".
    ___________parameters___________
    factors : list of strings 
    index   : desired string in factors to omit from product
    '''
    prod = ""
    for i, factor in enumerate(factors): 
        if i != index:
            prod += "(" + factor + ")" + "*"
    prod = prod[:-1]
    return prod 

def readable_multiply_but_one(factors, index):
    '''
    Takes in a list of strings of the form 
        [ "(x - a_1)", "(x - a_2)", ..., "(x - a_n)"]
    and given i in {1, 2, ..., n}, multiply_but_one returns the string 
        _____
        |   |
        |   |  (x - a_j)
        j != i  
    Ex: factors = ["(x - 1)", "(x - 2)",  "(x - 3)", "(x - 4)"] 
        index   =  1
    would return  the string
        "(x - 1)(x - 3)(x - 4)".
    ___________parameters___________
    factors : list of strings 
    index   : desired string in factors to omit from product
    '''
    prod = 1
    for i, factor in enumerate(factors): # factor is a degree one polynomial (x - factor)
        if i != index:
            prod = polynomial.polymul( prod, factor)
    return prod

def loewner_matrix(x_data, y_data):
    assert len(x_data) == len(y_data), "input x and y data are mismatched"
    '''
        Computes the Loewner matrix defined as follows. 
        
        Left p = numer, q = denom.
        Consider 
            x_data = [x_1, x_2, ..., x_n]
        The function 
            1. Calculates y_data
            2. Creates a disjoint partitions of x_data
            [mu_1, ..., mu_{n_1}], [lambda_1, ..., lambda_{n_2}]
            and corresponding disjoint partitions of y_data 
                [v_1, ..., v_{n_1}], [w_1, ..., w_{n_2}]
            3. Calculates the Loewner matrix defined as 
                            v_i - w_j
                L_{ij} = ---------------
                        mu_i - lambda_j
            4. Calculates the smaller Loewner matrix 
        and returns the nullspace of this smaller matrix. 
        For more details, see Algorithm 1.1, P. 33 of [Cosmin-Ionita].

        ___________parameters___________
        x_data : list of numbers 
        numer  : numerator of function values we wish to interpolate 
        denom  : ""
        '''        
    mu_data, lambda_data = split_by_alternating(x_data)
    v_data, w_data = split_by_alternating(y_data)
    n = len(mu_data)
    m = len(lambda_data)

    L = np.zeros(n*m).reshape(n, m)
    for i, v_i in enumerate(v_data):
        for j,  w_j in enumerate(w_data):
            val = (v_i - w_j)/(mu_data[i]  - lambda_data[j])
            L[i,j] = val

    '''
    We now work on calculating the smaller Loewner matrix
    '''
    rank =  matrix_rank(L)
    L_hat = np.zeros( (n + m - (rank + 1))*(rank + 1) ).reshape(n + m - (rank + 1), rank + 1)
    lambda_hat, mu_hat = split_in_half(x_data, index = rank + 1)
    w_hat, v_hat = split_in_half(y_data, index = rank + 1)
    
    for i, v_i in enumerate(v_hat):
        for j,  w_j in enumerate(w_hat):
            val = (v_i - w_j)/(mu_hat[i]  - lambda_hat[j])    
            L_hat[i, j] = val
    return L, L_hat, null_space(L_hat),lambda_hat, w_hat 

'''
Example from the paper 
x_data = list(np.linspace(-1, 1,  21))
p = "50*x^2 + 13"
q = "(50*x^2 -50*x +13)*(50*x^2 + 50*x +13)"
y_data = p_over_q_vals(p, q, x_data)
Then call Loewner. The nullspace gives us the coefficients.
'''

def loewner_polynomial(nullspace, lambda_hat, w_hat):
    '''
    Takes in the nullspace, lambda_hat, w_hat values and returns 
    the approximate polynomial that interpolates the data. 
    
    ___________parameters___________
    nullspace  : a numpy array containing the basis of the nullspace
    lambda_hat : list of values 
    w_hat      : list of values 
    (See [Cosmin-Ionita], P.33)
    '''
    nullspace = [x[0] for x in nullspace]
    factors = []
    for lamb in lambda_hat:
        factors.append("x -  " + str(lamb))

    #Initialize the numerator and denominator
    factor = multiply_but_one(factors, 0)
    numer = " " + str(nullspace[0]) + "*" + str(w_hat[0]) + "*(" + factor + ")" 
    denom = " " + str(nullspace[0]) + "*(" + factor + ")"

    for i in range(1, len(lambda_hat)):
        factor = multiply_but_one(factors, i)
        numer = " " + str(nullspace[i]) + "*" + str(w_hat[i]) + "*(" + factor + ")" + "+" + numer 
        denom = " " + str(nullspace[i]) + "*(" + factor + ")" + "+" + denom
    return numer, denom

def readable_loewner_polynomial(nullspace, lambda_hat, w_hat):
    '''
    Takes in the nullspace, lambda_hat, w_hat values and returns 
    the approximate polynomial that interpolates the data. 
    
    ___________parameters___________
    nullspace  : a numpy array containing the basis of the nullspace
    lambda_hat : list of values 
    w_hat      : list of values 
    (See [Cosmin-Ionita], P.33)
    '''
    nullspace = [x[0] for x in nullspace]
    factors = []
    for lamb in lambda_hat:
        factors.append((lamb, 1))

    numer = 0
    denom = 0

    for i in range(0, len(lambda_hat)):
        numer_summand = polynomial.polymul(readable_multiply_but_one(factors, i), nullspace[i]*w_hat[i])
        numer = polynomial.polyadd(numer_summand, numer)
        
        denom_summand = polynomial.polymul(readable_multiply_but_one(factors, i), nullspace[i])
        denom = polynomial.polyadd(denom_summand, denom)
    return numer, denom

def numpy_poly_to_expression(coeffs):
    poly_expression = str(coeffs[0])
    for deg in range(1, len(coeffs)):
        poly_expression = str(coeffs[deg]) + "x^" + str(deg) + " + " + poly_expression 
    return poly_expression

def rational_interpolate(x_data, y_data):
    '''
    Combines all of our previous work to return a rational function
    which interpolates the given (x, y) data. 

    ___________parameters___________
    x_data : list of numbers 
    numer  : numerator of function values we wish to interpolate 
    denom  : ""
    '''
    L, l_hat, a, l, w = loewner_matrix(x_data, y_data)
    numer, denom = loewner_polynomial(a, l, w)
    numer_exp, denom_exp = readable_loewner_polynomial(a, l, w)
    
    return numer, denom, numpy_poly_to_expression(numer_exp), numpy_poly_to_expression(denom_exp)
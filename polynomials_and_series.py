import numpy as np
import math

def series(terms, start = 0, end = None):
    '''
    Let 
        [a_0, a_1, ..., a_m] = terms
        j = start (default is j = 0).
        n = end
    This function computes the sum 
         n
        ----,
        \  
        /     a_j  =  a_j + a_{j+1} + ... + a_n.
        ----`
        i = j

    ___________parameters___________
    terms : (list of floats) elements of our series
    start : (int) index to begin sum; 
            **default value is zero**
    end   : (int) index to end the sum;
            **default value is length of terms**
    ''' 
    if end is None: # If not specified, assign default value
        end = len(terms)
    
    assert start < end, "start index needs to be less than end index"
    
    sum = 0 # begin the sum 
    for i in range(start, end):
        sum += terms[i]
    return sum

def p_over_q_vals(p, q, terms, 
                  start =  0,
                  end   =  None, 
                  coeff = 1, 
                  scale = 1,):
    '''
    Let p, q be functions. Let c be a constant. 
    Then p_over_q_vals computes the list 

        ___                                               ___
        |          p(0)          p(1)                 p(n)   |
        |   (c)^0 ------  (c)^1 ------         (c)^n ------  |
        |          q(0) ,        q(1)  , ... ,        q(n)   |
        ___                                               ___
        x = 0
    In our case, these functions will be polynomials, but they don't have to be. 

    ___________parameters___________
    p     : (string) numerator function   Ex: 10*x^8 + 13*x^2 + 3
    q     : (string) denominator function 
    terms : (list of floats) terms which need to be evaluated
    coeff : (float) constant c (see equation)
            **Default value is 1**
    scale : (float) scalar to scale the sum
            **Default value is 1**
    '''
    if end is None: # If not specified, assign default value
        end = len(terms)
    
    assert start < end, "start index needs to be less than end index"

    vals = []  # where we store the values of the series 
    p = p.replace("^", "**") #** We let the user write ^ instead of ** because ** is annoying
    q = q.replace("^", "**")

    for term in terms:# + 1 to make sure end is included
        val_p = p.replace("x","(" +  str(term) + ")")
        val_q = q.replace("x","(" +  str(term) + ")")
        val = scale * ((coeff)**term) * (eval(val_p)/eval(val_q))
        vals.append(val)
    return vals

def poly_str_expr(deg, coeffs, nopars = False):
    '''
    Creates readble polynomial expressions. 
    Ex:
        >>> poly_str_expr(4, [1, 2, 3, 4, 5], nopars = True)
        "1*x^4 + 2*x^3 + 3*x^2 + 4*x^1 + 5"

    ___________parameters___________
    deg    : (int) degree of output polynomial 
    coeffs : (list of floats) (From left to right) list of coefficients of the polynomial 
    nopars : (boolean) set True to remove parentheses (for readability)
             **Default value is False (for computations)** 
    '''
    if deg != len(coeffs)-1:
        raise ValueError("Coefficient inputs and degree mismatched", deg, len(coeffs), coeffs)

    expr = str(coeffs[-1]) # We first add the constant terms.
    coeffs.pop(-1)         # Drop it, so that we don't have to tip-toe around it.

    for i, coeff in reversed(list(enumerate(coeffs))):
        expr =  "(" + str(coeff) + ")" \
            + "*x^(" + str(deg - i) + ") + "\
            + expr
    if nopars == True:
        expr = expr.replace("(", "").replace(")","")
    return expr

def p_over_q_expr(num_deg, den_deg, params, nopars = False):
    '''
    Creates readable rational function expressions.
    Let 
        num_deg = n,
        den_deg = m,    
        params = [c_n, c_{n-1}, ... , c_0, b_m, b_{m-1}, ..., b_0].
    Then this function creates string expression for the polynomials
        p(x) = c_nx^n + c_{n-1}^{n-1} + ... + c_0
        q(x) = b_mx^m + b_{m-1}^{m-1} + ... + b_0.
    
    ___________parameters___________
    num_deg : (int) degree of numerator 
    den_deg : (int) degree of denominator
    params  : (list of floats) our coefficients. See above for ordering.
    no pars : (boolean) set True if we want to eliminate parenthesis
              **Default value is False**
    '''
    if num_deg + den_deg != len(params)-2:
        raise ValueError("Coefficient-degree mismatch", num_deg, den_deg, params)
    p_coeffs = params[0:num_deg+1]
    q_coeffs = params[1+ num_deg:]

    p_expr = poly_str_expr(num_deg, p_coeffs, nopars)
    q_expr = poly_str_expr(den_deg, q_coeffs, nopars)
    return p_expr, q_expr
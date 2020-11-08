import numpy as np
import math
from scipy.optimize import curve_fit
from decimal import *
getcontext().prec = 40
'''
We look for a BBP base 10 series for pi. A desired formula is of the form 
pi \sum_{k = 0}(1/10)^k * p(k)/q(k)
where p, q are polynomials in k with integer coefficients. 
This might not even exist. Borwein et al showed that a class of series, which can be rewritten 
in BBP form, cannot express pi with base 10. Thus, if it exists, it needs to be something 
else, although they are a bit vague in their papers on this. Nevertheless, this 
aims to find such an expression for pi.

Part of this problem boils down to interpolating data with rational functions. 
Such interpolation is hard, and unforunately not much progess has 
been made on rational function interpolation since the 1900s. I can nevertheless build 
a cheap one. 
'''


# For testing
five_pi = 3.14159    
ten_pi = 3.1415926535


def series(terms, start = 0, end = None):
    '''
    Let n = end
        j = start.
    This function computes the sum 
         n
        ----,
        \  
        /     a_j  =  a_j + a_{j+1} + ... + a_n.
        ----`
        i = j
    where terms = [a_0, a_1, ..., a_m], m >= n. 

    ___________parameters___________
    terms : list of numbers
    start : index to begin sum; 
            **default value is zero**
    end   : index to end the sum;
            **default value is length of terms**
    ''' 
    if end is None: # If not specified, assign default value
        end = len(terms)
    
    assert start < end, "start index needs to be less than end index"
    
    sum = 0 # begin the sum 
    for term in range(start, end + 1): # + 1 to make sure end is included
        sum += terms[i]
    return sum

def p_over_q_vals(p, q, terms, 
                  start =  0,
                  end   =  None, 
                  coeff = 1, 
                  scale = 1,):
    '''
    Let p, q be functions. Let c be a constant. 
    Then p_over_q_vals computes the sum 
          n
        -----,
        \      
         \          p(x)           p(0)           p(1)                 p(n)
         /   (c)^x ------ = (c)^0 ------ + (c)^1 ------ + ... + (c)^n ------
        /           q(x)           q(0)           q(1)                 q(n)
        -----`  
        x = 0
    In our case, these functions will be polynomials, but they don't have to be. 

    ___________parameters___________
    p     : numerator function   Ex: 10*x^8 + 13*x^2 + 3
    q     : denominator function 
    coeff : constant c (see equation)
            **Default value is 1**
    scale : scalar to scale the sum
            **Default value is 1**
    '''
    if end is None: # If not specified, assign default value
        end = len(terms)
    
    assert start < end, "start index needs to be less than end index"


    vals = []  # where we store the values of the series 
    p = p.replace("^", "**") #** We let the user write ^ instead of ** because ** is annoying
    q = q.replace("^", "**")

    for term in range(start, end +1):# + 1 to make sure end is included
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
    deg    : degree of output polynomial 
    coeffs : (From left to right) list of coefficients of the polynomial 
    nopars : Boolean, set True to remove parentheses (for readability)
             **Default value is False (for computations)** 
    '''
    assert deg == len(coeffs) - 1, "Coefficient-degree mismatch"

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
    num_deg : degree of numerator 
    den_deg : degree of denominator
    params  : list containing our coefficients. See above for ordering.
    no pars : set True if we want to eliminate parenthesis
              **Default value is False**
    '''
    assert num_deg + den_deg == len(params)-2,  "Coefficient-degree mismatch"
    p_coeffs = params[0:num_deg+1]
    q_coeffs = params[1+ num_deg:]

    p_expr = poly_str_expr(num_deg, p_coeffs, nopars)
    q_expr = poly_str_expr(den_deg, q_coeffs, nopars)
    return p_expr, q_expr


############ We need a global variable (see p_over_q_at_x). 
degs = [0,0]
############

def p_over_q_at_x(x, *params):
    '''
    Helper for grad_descent_BBP_rational.
    Let x be a real number. 
        p(x) = c_nx^n + c_{n-1}^{n-1} + ... + c_0
        q(x) = b_mx^m + b_{m-1}^{m-1} + ... + b_0.
    This function simply evaluates 
        p(x)
        ----
        q(x)
    at x. 
    ___________parameters___________
    x      : the value we wish to evaluate our rational polynomial with 
    params : the coefficents (comma separated).
    '''
    num_deg = degs[0] # we access our global variable 
    den_deg = degs[1]
    params = list(params) # we place our coefficients  in a list before passing it off
    p_expr, q_expr = p_over_q_expr(num_deg, den_deg, params)

    p_expr = p_expr.replace("^", "**") # clean up for calculation
    q_expr = q_expr.replace("^", "**")
    return eval(p_expr)/eval(q_expr)   # evaluate and return 

def grad_descent_BBP_rational(num_deg, den_deg,
                              func_to_fit_num = "(0.625)^(x)*(120*x^2 + 151*x + 47)", 
                              func_to_fit_den = "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15",
                              num_pts = None,
                              x_data = None,
                              guess = None, 
                              output_params = False):
    '''
    Let n = num_deg, 
        m = den_deg. 
        func_to_fit_num = f(x)
        func_to_fit_den = g(x).
    This function attempts to find parameters such that the function 
           c_nx^n + c_{n-1}^{n-1} + ... + c_0
        ------------------------------------------
           b_mx^m + b_{m-1}^{m-1} + ... + b_0.
    models f(x)/g(x) on x_data as closely as posisble. 

    ___________parameters___________
    num_deg : the degree of the numerator polynomial we would like to fit. 
    den_deg : the degree of the denominator...
    func_to_fit_num : numerator of function we want to model 
                      **By default, this is the BBP numerator**
    func_to_fit_den : denominator of function we want to fit
                      **By default, this is the BBP denominator** 
    x_data : The exact x_data points we want to compare 
             **By default, x_data = [0, 1, 2, ... , n_terms]**
    nunm_pts      : If x_data is default, we specify the number of integers we want to compare 
                    **No default value; should only be set when x_data is not set**
    guess         : our guess of what the coefficients should be (can just use a graph)
                    **By default, guess = [1, 1, ..., 1] (we add the appropriate number of ones)**
    output_params : True if want a list of the approximated coefficients
                    **By default, this is False (it's an unreadable mess 
                    but can be set to True to pass to other functions)**
    '''
    if guess is None:  # Check if guess was overwritten
        guess = [1]*(num_deg + den_deg + 2)
    if x_data is None: # Check if x_data was overwritten
        x_data = x_data = np.linspace(0, n_terms, n_terms+1, dtype=np.longdouble)
    # We change the global parameter
    global degs 
    degs = [num_deg, den_deg]
    
    # Obtain our function valuesthat we will try to fit
    func_vals = np.array(p_over_q_vals(func_to_fit_num, func_to_fit_den, n_terms), dtype=np.longdouble)
    
    # We now calculate our parameters to try to fit. Obtain polynomial expressions.
    params, cov = curve_fit(p_over_q_at_x, x_data, func_vals, p0=guess, maxfev=5000) 
    p, q = p_over_q_expr(num_deg, den_deg, list(params), nopars = True) 

    # Calculate their values on the x_data. We also calculate the error.
    approxed_vals = p_over_q_vals(p, q, n_terms)
    error = list(map(lambda x,y : abs(x - y), func_vals, approxed_vals))
    total_error = sum(error)

    #We print the data for the user.
    print("For func :  ", func_to_fit_num + "\n",
          "Numerator:  ", p.replace("*", "")+"\n"\
          "Denominator:", q.replace("*", "")+"\n"\
          "Error:      ", total_error, "\n") # for readability
    if output_params:
        return params, total_error
    else:
        return p.replace("*", ""), q.replace("*", ""), total_error


def gradient_recursion(n_iters, num_deg, den_deg, init_fit, 
                       step = 100
                       num = "(0.625)^(x)*(120*x^2 + 151*x + 47)", 
                       den = "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15",
                       zero = True):
    '''
    Theoretically, we are trying to fit infinitely many values with a rational function. 
    Thus to do this, we first fit the first 100 values, then use the estimated parameters as a guess 
    for a fit of 1000 values, then use those estimated parameters for the next fit, and so on

    n_iters  : The number of values of points from the data that we would like to fit 
    num_deg  : Degree of numerator we think will fit 
    den_deg  : ""
    num      : Numerator of function we want to fit 
    den      : ""
    init_fit : The number of points we will try to fit in our first step of the algorithm.
    step     : The number of points to add to the next approximation in each loop
    '''
    # Initial conditions
    try_guess = [1]*(num_deg + den_deg + 2) # Initial guess (just ones)
    iters = init_fit                        # Initial # of points to fit
    prev_error = 0                          # Initial error

    # Algorithm loop
    while iters < n_iters:
        print("Iteration: ", iters) 
        '''
        Idea:
        We use the previously-fitted coefficients for our next fit and repeat 
        while we gradually increase the number of points of the function 
        that we are trying to model. So, our parameters get more and more 
        accurate through each iteration. 
        '''
        params, error = grad_descent_BBP_rational(iters,\
                                            num_deg, den_deg,\
                                            func_to_fit_num = num,\
                                            func_to_fit_den = den,\
                                            guess = try_guess,\
                                            output_params = True,\
                                            include_zero = zero)
        try_guess = params
        iters += step
    # We're finished looping, so obtain the polynomials for our super accurate approximation
    p, q = p_over_q_expr(num_deg, den_deg, list(try_guess), nopars = True) 
    return p.replace("*", ""), q.replace("*", ""), error

def search_and_compare():
    '''
    In all of our previous functions, we assumed the user knew the numerator and
    denominator degree of the rational function that best fits their data. 
    That isn't always the case. 

    This loops through all possible rational function degree combinations for the numerator and
    denominator. One by one, it prints to the user the error of each approximation. 
    It also collects the best approximations in best_approxs. 
    '''

    exp_fits = []
    bbp_fits = []
    best_approxs = []

    for num_deg in range(1, 9):
        for den_deg in range(2, 12):
            print(num_deg, den_deg)
            try:
                p, q, error = gradient_recursion(100000, num_deg, den_deg,\
                                                        func_to_fit_num = "(0.625)^x", \
                                                        func_to_fit_den = "1",
                                                        include_zero = zero)
                data = [p, q, error]
                if error < 1e-7:
                    best_approxs.append(data)
                exp_fits.append(data)
            except:
                pass
            try:
                p, q, error = gradient_recursion(100000, num_deg, den_deg, include_zero = zero)
                data = [p, q, error]
                if error < 1e-7:
                    best_approxs.append(data)
                bbp_fits.append(data)
            except: 
                pass
    return exp_fits, bbp_fits, best_approxs




















############ For later 
def brute_rational_search(n_runs, delta, n_comparisons):
    '''
    This is inefficient gradient descent, minimizing sum of errors. 
    I don't want to use least squares because my data is very, very small.

    We cycle through our parameters over and over again, tweaking them 
    slightly to minimize error.

    PARAMETERS:
    n_runs : number of times we cycle to tweak parameters
    delta  : amount we change each coefficient during each step of tweaking 
    n_comparisons : number of BBP data points we try to fit with our  function.
    '''
    scaled_bbp_vals = p_over_q("120*x^2 + 151*x + 47", "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15", n_comparisons, 10/16)
    p = "{A}*x^2 + {B}*x + {C}"
    q = "{D}*x^5 + {E}*x^4 + {F}*x^3 + {G}*x^2 + {H}*x^1 + {I}" 
    #### Initial conditions #####
    a = 22
    b = 151
    c = 47
    d = 80
    e = 600
    f = 1000
    g = 800
    h = 200
    i = 15
    
    params = {"A" : a, "B": b, "C":c, "D" : d, "E" : e, "F" : f, "G" : g, "H" : h, "I" : i}
    numer_func = p.format(**params)
    denom_func = q.format(**params)
    vals = p_over_q(numer_func, denom_func, n_comparisons, 1)
    prev_diffs = list( map(lambda x,y : abs(x - y), scaled_bbp_vals, vals) )
    ############################

    run = 0
    while True:    
        if run % (n_runs/10) == 0:
            print(round(run/n_runs,2) * 100, "percent ...")
        for coeff in params:
            params[coeff] += delta
            plus_numer = p.format(**params)
            plus_denom = q.format(**params)
            plus_vals = p_over_q(plus_numer, plus_denom, n_comparisons, 1)
            plus_diffs = list(map(lambda x,y : abs(x - y), scaled_bbp_vals, plus_vals))
            plus_error = sum(plus_diffs)

            params[coeff] -= 2*delta # we do 2* to offset the added delta from the above step 
            sub_numer = p.format(**params)
            sub_denom = q.format(**params)
            sub_vals = p_over_q(sub_numer, sub_denom, n_comparisons, 1)
            sub_diffs = list(map(lambda x,y : abs(x - y), scaled_bbp_vals, sub_vals))
            sub_error = sum(sub_diffs)
            ''' 
            (1) At this point, params[coeff] is a delta less than its original value. 
            '''
            if sub_error <= plus_error: # then sub_error gets us closer
                if not sub_error <= sum(prev_diffs): # if it is worse
                    params[coeff] += delta  # return parameter to its original value
                else:
                    prev_diffs = sub_diffs
            else: # then plus_error gets us closer
                if not plus_error <= sum(prev_diffs): #if it is wosse
                    params[coeff] += delta 
                else:
                    # then plus error was good, so return parameter to one delta above its original value.
                    params[coeff] += 2*delta 
                    prev_diffs = plus_diffs
            # we loop, and update our polynomials
            numer_func = p.format(**params)
            denom_func = q.format(**params)
        if run == n_runs:
            print("Numerator:  ", numer_func.replace("*", "").replace("k","x")+"\n"\
                  "Denominator:", denom_func.replace("*", "").replace("k","x")+"\n"\
                  "Total error:", sum(prev_diffs))
            return prev_diffs
        run += 1

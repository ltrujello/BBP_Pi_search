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

def series(terms, n_terms):
    '''
    sums the terms up to n_terms.
    '''
    sum = 0
    for i in range(0, n_terms):
        sum += terms[i]
    return sum

def p_over_q_vals(p, q, n_terms, coeff = 1, scale = 1):
    '''
    Computes the individual terms of a series with rational function p(k)/q(k) with 
    coefficient coeff.
    '''
    vals = []
    p = p.replace("^", "**") #** is annoying to write for exponentiation
    q = q.replace("^", "**")
    for term in range(0, n_terms+1):
        val_p = p.replace("x","(" +  str(term) + ")")
        val_q = q.replace("x","(" +  str(term) + ")")
        val = scale * ((coeff)**term) * (eval(val_p)/eval(val_q))
        vals.append(val)
    return vals

def bbp_rationals():
    '''
    We compute the terms of the series given by the rationals in the BBP formula for pi.
    '''
    rationals = []
    numer_val = p_over_q_vals("120*x^2 + 151*x + 47", "1", 50, 5)
    denom_val = p_over_q_vals("512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15", "1", 50, 1)
    for i in range(0, len(numer_val)):
        rationals += [(numer_val[i], denom_val[i])]
    return rationals


def rational_series(p,q, n_terms, coeff = 1):
    '''
    To compute the total sum with terms (coeff)^k*p(k)/q(k)
    '''
    series_vals = p_over_q_vals(p, q, n_terms, coeff)
    return series(series_vals, n_terms)

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

'''
Let's try to instead use scipy gradient descent.
'''

def p_over_q_expr(num_deg, den_deg, params, nopars = False):
    '''
    If num_deg (numerator degree) = n,
       den_deg (denominator degree) = m,
    so that 
        params = [c_n, c_{n-1}, ... , c_0, b_m, b_{m-1}, ..., b_0].
    then our polynomials  are
        p(x) = c_nx^n + c_{n-1}^{n-1} + ... + c_0
        q(x) = b_mx^m + b_{m-1}^{m-1} + ... + b_0.
    The function creates string expressions of p(x) and q(x). 

    num_deg : degree of numerator 
    den_deg : ""
    params  : list containing our coefficients. See above for ordering;  we 
              basically read from left to right, starting from top to bottom.
    no pars : set True if we want to eliminate parenthesis (e.g., for readibility). 
    '''
    assert num_deg + den_deg == len(params)-2,  "Coefficient-degree mismatch"
    p_coeffs = params[0:num_deg+1]
    q_coeffs = params[1+ num_deg:]

    p_expr = str(p_coeffs[-1]) # We first add the constant terms.
    p_coeffs.pop(-1) # Drop it, so that we don't have to tip-toe around it or accidentally add it later.
    q_expr = str(q_coeffs[-1])
    q_coeffs.pop(-1)

    for i, coeff in reversed(list(enumerate(p_coeffs))):
        p_expr =  "(" + str(coeff) + ")" \
            + "*x^(" + str(num_deg - i) + ") + "\
            + p_expr
    for i, coeff in reversed(list(enumerate(q_coeffs))):
        q_expr =  "(" + str(coeff) + ")" \
            + "*x^(" + str(den_deg - i) + ") + "\
            +  q_expr
    if nopars == True:
        p_expr = p_expr.replace("(", "").replace(")","")
        q_expr = q_expr.replace("(", "").replace(")","")
    return p_expr, q_expr


########## We need a global variable (see p_over_q_at_x). 
degs = [0,0]
#########################################################

def p_over_q_at_x(x, *params):
    '''
    Helper for grad_descent_BBP_rational.
    With x a real number, params containing the coefficients of our 
    rational function p(x)/q(x), we evalute the function at x and return 
    the value (See below for more detail).

    The reason why I need degs to be global is because it cannot be an argument 
    of this function. Yet, at the same time, I need it to be! Thus I set it oustide 
    and call it inside. It changes whenever I need to change it based on via 
    the ``global`` python keyword.
    '''

    num_deg = degs[0]
    den_deg = degs[1]
    params = list(params)
    '''
    (More detail.)
    If num_deg (numerator degree) = n,
       den_deg (denominator degree) = m,
    so that 
        params = [c_n, c_{n-1}, ... , c_0, b_m, b_{m-1}, ..., b_0].
    then our polynomials  are
        p(x) = c_nx^n + c_{n-1}^{n-1} + ... + c_0
        q(x) = b_mx^m + b_{m-1}^{m-1} + ... + b_0.
    The function evaluates p(x)/q(x) at x.
    '''
    p_expr, q_expr = p_over_q_expr(num_deg, den_deg, params)
    p_expr = p_expr.replace("^", "**")
    q_expr = q_expr.replace("^", "**")
    return eval(p_expr)/eval(q_expr)

sentinel = object() # dummy object
def grad_descent_BBP_rational(n_terms, num_deg, den_deg,
                              func_to_fit_num = "(0.625)^(x)*(120*x^2 + 151*x + 47)", 
                              func_to_fit_den = "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15",
                              guess = sentinel,
                              output_params=False):
    '''
    Let n = num_deg, m = den_deg. This function attempts to model a rational 
    function p(x)/q(x) with 
        p(x) = c_nx^n + c_{n-1}^{n-1} + ... + c_0
        q(x) = b_mx^m + b_{m-1}^{m-1} + ... + b_0.
    to the BBP data.
    
    n_terms : number of pts from BBP data we attempt to model.
    num_deg : the degree of the numerator polynomial we would like to fit. 
    den_deg : the degree of the denominator...
    guess   : our guess of what the coefficients should be (can just use a graph).
    '''
    #p0 = [22,151,150, 47,80,600,1000,800,200,100,15] a very good guess when n = 2, m = 5.
    if guess is sentinel:
        guess = [1]*(num_deg + den_deg + 2)
    global degs 
    degs = [num_deg, den_deg]
    # set up x, y data
    k_indices = np.linspace(0, n_terms, n_terms+1)
    func_vals = np.array(p_over_q_vals(func_to_fit_num, func_to_fit_den, n_terms))

    params, cov = curve_fit(p_over_q_at_x, k_indices, func_vals, p0=guess, maxfev=5000) # gradient descent
    p, q = p_over_q_expr(num_deg, den_deg, list(params), nopars = True) # obtain the expr for the new approximation

    approxed_vals = p_over_q_vals(p, q, n_terms)
    error = list(map(lambda x,y : abs(x - y), func_vals, approxed_vals))
    total_error = sum(error)
    print("For func :  ", func_to_fit_num + "\n",
          "Numerator:  ", p.replace("*", "")+"\n"\
          "Denominator:", q.replace("*", "")+"\n"\
          "Error:      ", total_error, "\n") # for readability
    if output_params:
        return params, total_error
    else:
        return p.replace("*", ""), q.replace("*", ""), total_error

def search_and_compare():
    exp_fits = []
    bbp_fits = []
    best_approxs = []

    for num_deg in range(1, 8):
        for den_deg in range(10, 11):
            print(num_deg, den_deg)
            try:
                p, q, error = gradient_recursion(100000, num_deg, den_deg,\
                                                        func_to_fit_num = "(0.625)^x", \
                                                        func_to_fit_den = "1")
                data = [p, q, error]
                if error < 1e-7:
                    best_approxs.append(data)
                exp_fits.append(data)
            except:
                pass
            try:
                p, q, error = gradient_recursion(100000, num_deg, den_deg)
                data = [p, q, error]
                if error < 1e-7:
                    best_approxs.append(data)
                bbp_fits.append(data)
            except: 
                pass
    return exp_fits, bbp_fits, best_approxs

# '5367124765888.0x^4 + -530171538689464.0x^3 + 1.7412531032379454e+16x^2 + -1.924913929564134e+17x^1 + 8871917683451926.0
# -13176492198746.0x^9 + 229588042882762.0x^8 + -4126924029947049.0x^7 + 18717880079013420x^6 + -184346450804085088x^5 + 663253966578081.0x^4 + -1637793963428539904x^3 + -26477463167523488x^2 + -230866126471856896x^1 + 2831463090463394.0

# p = [10734249531776, -1060343077378928, 34825062064758908, -384982785912826816, 17743835366903852]
# q = [-26352984397492, 459176085765524, -8253848059894098, 37435760158026840, -368692901608170176, 1326507933156162, -3275587926857079808, -52954926335046976, -461732252943713792, 5662926180926788]
# "0.0010734249531776873*x^4 + -0.10603430773789276*x^3 + 3.482506206475891*x^2 + -38.498278591282684*x^1 + 1.7743835366903853"
# "-0.002635298439749254*x^9 + 0.04591760857655244*x^8 + -0.8253848059894098*x^7 + 3.743576015802684*x^6 + -36.869290160817016*x^5 + 0.13265079331561627*x^4 + -327.558792685708*x^3 + -5.295492633504698*x^2 + -46.17322529437138*x^1 + 0.5662926180926788"
# "0.0012544894281814108*x^5 + -0.1180229478194571*x^4 + 3.7208845645745106*x^3 + -39.617001666980315*x^2 + -1.7661414102809467*x^1 + 1.6540985067121214"
# "-0.0021822819261304856*x^10 + 0.032424363138435705*x^9 + -0.624593501436639*x^8 + 2.01486859075354*x^7 + -28.267048159977428*x^6 + -32.6610635031047*x^5 + -279.16296146201023*x^4 + -101.22005231035284*x^3 + 1.2780562045517494*x^2 + -8.501259828766722*x^1 + 0.527903778737912"
'''
Here's an idea. What if we fit the function for the first ~50 BBP or (10/16)^x points, then 
use those coeffs for fitting ~100, and so on? 
'''

def gradient_recursion(n_iters, num_deg, den_deg,
                       num = "(0.625)^(x)*(120*x^2 + 151*x + 47)", 
                       den = "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15"):
    try_guess = [1]*(num_deg + den_deg + 2)
    iters = 100
    while iters < n_iters:
        print("Iteration: ", iters)
        params, error = grad_descent_BBP_rational(iters,\
                                            num_deg, den_deg,\
                                            func_to_fit_num = num,\
                                            func_to_fit_den = den,\
                                            guess = try_guess,\
                                            output_params = True)
        try_guess = params
        if iters < 2000:
            iters += 200
        else:
            iters += 10000
    p, q = p_over_q_expr(num_deg, den_deg, list(try_guess), nopars = True) # obtain the expr for the new approximation
    return p.replace("*", ""), q.replace("*", ""), error



'''
Sci_py gradient descent is very efficient, but lacks precision past ~12 decimal points. 
It is difficult to control precision in python since python was not made to do so. It is also 
difficult to control precision in numpy and it's kind of problematic at the moment (e.g., it 
misleads the user on what its np.longdouble actually is). 
'''

def grad_descent_for_coeffs(n_comparisons): 
    rows = []
    bbp_vals = np.array(p_over_q_vals("120*k^2 + 151*k + 47", "512*k^4 + 1024*k^3 + 712*k^2 + 194*k + 15",n_comparisons, 10/16))

    for k in range(0, n_comparisons):
        nth_row = []
        a_k = bbp_vals[k]
        expr = ["({k})**2", "{k}", "1", "-{a_k}*({k})**5", "-{a_k}*({k})**4", "-{a_k}*({k})**3", "-{a_k}*({k})**2", "-{a_k}*({k})", "-{a_k}"]
        for term in expr:
            term = term.format(a_k = a_k, k = k)
            nth_row.append(eval(term))  
        rows.append(nth_row)
    A = np.array(rows)
    B = np.zeros(n_comparisons)
    print(A.shape,B.shape)
    return A,B, bbp_vals
    # return A,B, np.linalg.solve(A,B)

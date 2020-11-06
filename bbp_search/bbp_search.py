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

sentinel = object() # dummy val
def grad_descent_BBP_rational(n_terms, num_deg, den_deg,
                              func_to_fit_num = "120*x^2 + 151*x + 47", 
                              func_to_fit_den = "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15",
                              guess = sentinel):
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
    print("Numerator:  ", p.replace("*", "")+"\n"\
          "Denominator:", q.replace("*", "")+"\n"\
          "Error:      ", sum(error)) # for readability
    return p.replace("*", ""), q.replace("*", ""), sum(error)

def search_and_compare():
    exp_fits = []
    bbp_fits = []
    best_approxs = []

    for num_deg in range(1, 7):
        for den_deg in range(1, 10):
            try:
                p, q, error = grad_descent_BBP_rational(100000, num_deg, den_deg,\
                                                        func_to_fit_num = "(0.625)^x", \
                                                        func_to_fit_den = "1")
                data = [p, q, error]
                if error < 1e-5:
                    best_approxs.append(data)
                exp_fits.append(data)
            except:
                pass
            try:
                p, q, error = grad_descent_BBP_rational(100000, num_deg, den_deg)
                data = [p, q, error]
                if error < 1e-5:
                    best_approxs.append(data)
                bbp_fits.append(data)
            except: 
                pass
        return exp_fits, bbp_fits
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



# By testing, the summands cannot by of the form
# p(x) = c_1x^2 + c_2x + c_3
# q(x) = b_1x^6 + b_2x^5 + b_3x^4 + b_4x^3 + b_5x^2 + b_6x + b_7

# Pretty close rational functions:
# 1)  6.900000000000155x^2 + 164.29999999999924x + 47.0 
#     / 95.09999999999914x^5 + 600.3000000000001x^4 + 986.4999999999969x^3 + 800.3000000000001x^2 + 200.29999999999998x + 15.0
# 2)  4.829999999998592x^2 + 166.5099999999859x + 47.00000000007106
#     113.60999999992283x^5 + 567.2550000000298x^4 + 1001.6399999999985x^3 + 800.8349999999992x^2 + 200.83499999999924x + 15.000000000017765
# 3) 16.94999999999993x^2 + 154.35000000000124x + 46.99999999999928
#    85.04999999999971x^5 + 600.6499999999944x^4 + 996.5499999999935x^3 + 800.7499999999944x^2 + 200.6500000000014x + 15.00000000000018    
# Really close!:
# 4)  11.950000000000001x^2 + 159.35000000000238x + 46.99999999999882
#     90.04999999999943x^5 + 600.6499999999887x^4 + 995.0499999999874x^3 + 800.7499999999887x^2 + 197.15000000000302x + 15.000000000000357    
# 5)  6.950000000000091x^2 + 164.35000000000352x + 46.999999999998465
#     95.04999999999914x^5 + 600.649999999983x^4 + 995.0499999999818x^3 + 800.7499999999831x^2 + 192.15000000000472x + 15.000000000000535
# 6)  9.474999999999394x^2 + 161.87500000000247x + 46.975
#     92.52500000000285x^5 + 599.7750000000002x^4 + 1000.7249999999993x^3 + 800.7749999999993x^2 + 190.17499999999777x + 14.975 
# 7)  6.974999999999395x^2 + 164.37500000000304x + 46.975
#     95.02500000000342x^5 + 599.7750000000002x^4 + 1000.7249999999993x^3 + 800.7749999999993x^2 + 187.6749999999972x + 14.975
# 8)  5.374999999999445x^2 + 165.9750000000034x + 46.975
#     96.57500000000404x^5 + 599.7750000000002x^4 + 1000.7249999999993x^3 + 800.7749999999993x^2 + 186.12499999999685x + 14.975
# 9)  12.392000000000259x^2 + 158.94400000001266x + 46.984000000004414
#     89.60799999999468x^5 + 600.7760000000665x^4 + 1000.7760000000665x^3 + 792.0560000000868x^2 + 200.79200000001663x + 14.999999999998854
#10)  3.35000000000007x^2 + 167.95000000000616x + 47.09999999999776
#     96.74999999999905x^5 + 600.7499999999717x^4 + 995.2499999999704x^3 + 800.8499999999717x^2 + 190.55000000000766x^1 + 15.000000000000774  
# Insanely close:
#11) 0.7912612958548907x^2 + -33.37646326882975x + 358.73514782417425
#    3.381859148699356x^6 + -10.207350235822211x^5 + 224.43086733262686x^4 + 393.2003226706365x^3 + 2270.931165159218x^2 + 1035.7249781717808x + 114.48994079522934

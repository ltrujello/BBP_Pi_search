import numpy as np
import math
from decimal import *
getcontext().prec = 40
from scipy.optimize import curve_fit
'''
We now focus on a particular form of a rational function of interest, and apply our techniques to refine 
the parameters. 
'''

def p_over_q_vals(p, q, n_terms, coeff = 1, scale = 1, step = 1):
    '''
    Computes the individual terms of a series with rational function p(k)/q(k) with 
    coefficient coeff.
    '''
    vals = []
    p = p.replace("^", "**") #** is annoying to write for exponentiation
    q = q.replace("^", "**")
    for term in list(np.arange(0, n_terms + step, step)):
        val_p = p.replace("x","(" +  str(term) + ")")
        val_q = q.replace("x","(" +  str(term) + ")")
        val = scale * ((coeff)**term) * (eval(val_p)/eval(val_q))
        vals.append(val)
    return vals

def p_over_q_expr(num_deg, den_deg, params, nopars = False):
    '''
    >>> grad_descent_BBP_rational(1000,4,9,guess = [1]*15, output_params = True)
    For func :   (0.625)^(x)*(120*x^2 + 151*x + 47)
    Numerator:   0.002300222439035955x^4 + -0.21683391518404518x^3 + 6.85410029622493x^2 + -73.35693475528515x^1 + 1.48085245888863
    Denominator: -0.0041029803788999305x^9 + 0.06302631034391377x^8 + -1.2100784389047303x^7 + 4.3618275975794765x^6 + -56.14992485372326x^5 + -40.80301440809534x^4 + -555.3827617295477x^3 + -88.53580026669054x^2 + -69.28282300046119x^1 + 0.47261248687934765
    Error:       8.662144923766876e-09 

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
    # params.insert(0, "120")
    # params.insert(3, "512")
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
    # print(p_expr, q_expr)
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
        guess = [1]*(num_deg + den_deg + 2 - 0) # subtracted zero
    global degs 
    degs = [num_deg, den_deg]
    # set up x, y data
    k_indices = np.linspace(0, n_terms, 2*n_terms+1)
    func_vals = np.array(p_over_q_vals(func_to_fit_num, func_to_fit_den, n_terms, step = 0.5))
    # print(k_indices, func_vals)
    params, cov = curve_fit(p_over_q_at_x, k_indices, func_vals, p0=guess, maxfev=10000) # gradient descent
    p, q = p_over_q_expr(num_deg, den_deg, list(params), nopars = True) # obtain the expr for the new approximation

    approxed_vals = p_over_q_vals(p, q, n_terms, step=0.5)
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

def gradient_recursion(n_iters, num_deg, den_deg,
                       num = "(120*x^2 + 151*x + 47)", 
                    #    num = "(0.625)^(x)*(120*x^2 + 151*x + 47)", 
                       den = "512*x^4 + 1024*x^3 + 712*x^2 + 194*x + 15"):
    '''
    Theoretically, we are trying to fit infinitely many values with a rational function. 
    Thus to do this, we first fit the first 100 values, then use the estimated parameters as a guess 
    for a fit of 1000 values, then use those estimated parameters for the next fit, and so on

    n_iters : The number of values of points from the data that we would like to fit 
    num_deg : Degree of numerator we think will fit 
    den_deg : ""
    num     : Numerator of function we want to fit 
    den     : ""
    '''
    try_guess = [1]*(num_deg + den_deg + 2 - 0) # Subtracted zero
    iters = 100     # we first try to fit 1000 points 
    prev_error = 0  
    while iters < n_iters:
        print("Iteration: ", iters) 

        params, error = grad_descent_BBP_rational(iters,\
                                            num_deg, den_deg,\
                                            func_to_fit_num = num,\
                                            func_to_fit_den = den,\
                                            guess = try_guess,\
                                            output_params = True)
        try_guess = params
        iters += 100
    p, q = p_over_q_expr(num_deg, den_deg, list(try_guess), nopars = True) # obtain the expr for the new approximation
    return p.replace("*", ""), q.replace("*", ""), error

def search_4_9_deg(n_iters):
    try_guess = [1]*15 #Initial guess is just unital coefficients.
    iters = 30         # we first try to fit 10 points 
    prev_error = 0  
    '''
    With x^8 coefficient set to 19/330.
    Iteration:  364
    For func :   (0.625)^(x)*(120*x^2 + 151*x + 47)
    Numerator:   -0.0019378323454741057x^4 + 0.03661220372163172x^3 + 2.03259588931075x^2 + -45.32810061289457x^1 + 0.9781450289090063
    Denominator: -0.005311389821214237x^9 + 19/330x^8 + -0.8679303666005287x^7 + -0.0012709290446702693x^6 + -21.57334744015802x^5 + -107.54771981735557x^4 + -202.10058790739356x^3 + -209.97856819578868x^2 + 18.995261367623552x^1 + 0.3121739453964914
    Error:       6.501409960706873e-08 

    With the numerator constant term held to 47/15 and the denominator constant term held to 1
    Iteration:  405
    For func :   (0.625)^(x)*(120*x^2 + 151*x + 47)
    Numerator:   0.003347304533405038x^4 + -0.3156373792017806x^3 + 9.981020073615916x^2 + -106.90018972644894x^1 + 47/15
    Denominator: -0.005980084168276445x^9 + 0.09196158295908237x^8 + -1.7644940885106979x^7 + 6.375910644650743x^6 + -81.83223406177174x^5 + -58.76633043175199x^4 + -807.4658760495702x^3 + -122.97865650886236x^2 + -97.92019559661787x^1 + 1
    Error:       8.650574160737739e-09 

    With numerator x^4 set to one 
    For func :   (0.625)^(x)*(120*x^2 + 151*x + 47)
    Numerator:   1x^4 + -50.03960037947444x^3 + 647.5080722902519x^2 + -178.27324362099358x^1 + 2.109812151640777
    Denominator: 0.9225724662663526x^9 + -12.193671451032777x^8 + 161.67413737114924x^7 + -487.66918751431393x^6 + 3213.5970451701555x^5 + -2.3966908808525695x^4 + 2754.0897223022343x^3 + 701.0920035221563x^2 + -1109.1428130808786x^1 + 0.6733443037151416
    Error:       1.1607634827440615e-07 
    '''

    while iters < n_iters:
        print("Iteration: ", iters) 
        params, error = grad_descent_BBP_rational(iters,\
                                            4, 9,\
                                            guess = try_guess,\
                                            output_params = True)
        try_guess = params
        # if iters < 2000: # We speed up the algorithm after 2000 points. What's key is fitting the first ~2000 points.
        #     iters += 200   
        # else:
        iters += 1
    p, q = p_over_q_expr(4, 9, list(try_guess), nopars = True) # obtain the expr for the new approximation
    return p.replace("*", ""), q.replace("*", ""), error

def search_coeffs(terms, err, den, num):
    L = []
    for i in range(0, terms):
        var = (i/den)*num
        if abs(var - int(var)) < err:
            print(var)
            L.append(i)
    return L
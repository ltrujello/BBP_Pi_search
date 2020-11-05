import numpy as np
from scipy.optimize import curve_fit
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

def p_over_q_vals(p,q, n_terms, coeff = 1):
    '''
    Computes the individual terms of a series with rational function p(k)/q(k) with 
    coefficient coeff.
    '''
    vals = []
    p = p.replace("^", "**") #** is annoying to write for exponentiation
    q = q.replace("^", "**")
    for term in range(0, n_terms+1):
        val_p = p.replace("k","(" +  str(term) + ")")
        val_q = q.replace("k","(" +  str(term) + ")")
        # print(coeff**term)
        val = ((coeff)**term) * (eval(val_p)/eval(val_q))
        vals.append(val)
    return vals

def bbp_rationals():
    '''
    We compute the terms of the series given by the rationals in the BBP formula for pi.
    '''
    rationals = []
    numer_val = p_over_q("120*k^2 + 151*k + 47", "1", 50, 5)
    denom_val = p_over_q("512*k^4 + 1024*k^3 + 712*k^2 + 194*k + 15", "1", 50, 1)
    for i in range(0, len(numer_val)):
        rationals += [(numer_val[i], denom_val[i])]
    return rationals


def rational_series(p,q, n_terms, coeff = 1):
    '''
    To compute the total sum with terms (coeff)^k*p(k)/q(k)
    '''
    series_vals = p_over_q(p, q, n_terms, coeff)
    return series(series_vals, n_terms, coeff)

def brute_rational_search(n_runs, delta, n_comparisons):
    '''
    This is rational function interpolation. I have an idea on what the 
    form of my rational functions should be, so I am interpolating it to fit 
    the data I need to find my series for pi. 
    '''
    scaled_bbp_vals = p_over_q("120*k^2 + 151*k + 47", "512*k^4 + 1024*k^3 + 712*k^2 + 194*k + 15", n_comparisons, 10/16)
    p = "{A}*k^2 + {B}*k + {C}"
    q = "{D}*k^5 + {E}*k^4 + {F}*k^3 + {G}*k^2 + {H}*k^1 + {I}" 
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
Let's try to instead use gradient descent.
'''

def p_over_q(x, c_1, c_2, c_3, b_1, b_2, b_3, b_4, b_5, b_6):
    '''
    p(x) = c_1x^2 + c_2x + c_3
    q(x) = b_1x^5 + b_2x^4 + b_3x^3 + b_4x^2 + b_5x + b_6
    '''
    return (c_1*x**2 + c_2*x + c_3)/(b_1*x**5 + b_2*x**4 + b_3*x**3 + b_4*x**2 + b_5*x + b_6)

def approx_BBP_rational(n_terms):
    k_indices = np.linspace(0, n_terms, n_terms+1)
    scaled_bbp_vals = np.array(p_over_q_vals("120*k^2 + 151*k + 47", "512*k^4 + 1024*k^3 + 712*k^2 + 194*k + 15", n_terms, 10/16))
    params, cov = curve_fit(p_over_q, k_indices, scaled_bbp_vals, p0=[22,151,47,80,600,1000,800,200,15], method = "lm")
    params = {"A" : params[0], "B": params[1], "C": params[2], "D" : params[3], "E" : params[4], "F" : params[5], "G" : params[6], "H" : params[7], "I" : params[8]}
    p = "{A}*k^2 + {B}*k + {C}"
    q = "{D}*k^5 + {E}*k^4 + {F}*k^3 + {G}*k^2 + {H}*k^1 + {I}" 
    numer_func = p.format(**params)
    denom_func = q.format(**params)
    print("Numerator:  ", numer_func.replace("*", "").replace("k","x")+"\n"\
          "Denominator:", denom_func.replace("*", "").replace("k","x")+"\n")
 

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


def full_rational_search(): 
    '''
    Idiotic level of brute force; but may need later.
    '''
    p = "{A}*k^2 + 151*k + 47"
    q = "{C}*k^4 + {D}*k^3 + 712*k^2 + 194*k + 15" 
    c = 10/16
    for a in range(0, 1000):
        print("I'm ", a/10, " percent done!")
        for b in range(0, 1000):
            print("I'm b! ", b)
            for c in range(0, 1000):
                # print("I'm c!", c)
                numer_func = p.format(A = a, C = b, D = c)
                denom_func = q.format(A = a, C = b, D = c)
                vals = p_over_q(numer_func, denom_func, 50, c)
                for i, val in enumerate(vals):
                    matches = 0
                    if abs(scaled_bbp_vals[i] - val) < 1e-5:
                        matches+=1
                if matches > 10:
                    print(a,b,c)
                
'''
We look for a BBP base 10 series for pi. A desired formula is of the form 
pi \sum_{k = 0}(1/10)^k * p(k)/q(k)
where p, q are polynomials in k with integer coefficients. 
This might not even exist. Borwein et al showed that a class of series, which can be rewritten 
in BBP form, cannot express pi with base 10. Thus, if it exists, it needs to be something 
else, although they are a bit vague in their papers on this. 

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

def p_over_q(p,q, n_terms, coeff = 1):
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
    q = "{D}*k^5 + {E}*k^4 + {F}*k^3 + {G}*k^2 + {H}*k + {I}" 
    # Initial conditions. I found these just by looking at a graph. 
    a = 22
    b = 151
    c = 47
    d = 80
    e = 600
    f = 1000
    g = 800
    h = 200
    i = 15
    
    params = {"A" : a, "B": b, "C" : c, "D" : d, "E" : e, "F" : f, "G" : g, "H" : h, "I" : i}
    numer_func = p.format(**params)
    denom_func = q.format(**params)

    run = 0
    while True:    
        vals = p_over_q(numer_func, denom_func, n_comparisons, 1)
        diffs = list( map(lambda x,y : abs(x - y), scaled_bbp_vals, vals) )
        if run % 100 == 0:
            print(run//n_runs * 100, "percent ...")
        for param_key in params:
            params[param_key] += delta
            plus_numer = p.format(**params)
            plus_denom = q.format(**params)
            plus_vals = p_over_q(plus_numer, plus_denom, n_comparisons, 1)
            plus_diffs = list(map(lambda x,y : abs(x - y), scaled_bbp_vals, plus_vals))
            plus_error = sum(plus_diffs)

            params[param_key] -= 2*delta # we do 2* to offset the added delta from the above step 
            sub_numer = p.format(**params)
            sub_denom = q.format(**params)
            sub_vals = p_over_q(sub_numer, sub_denom, n_comparisons, 1)
            sub_diffs = list(map(lambda x,y : abs(x - y), scaled_bbp_vals, sub_vals))
            sub_error = sum(sub_diffs)
            ''' 
            (1) At this point, params[param_key] is a delta less than its original value. 
            '''
            if sub_error <= plus_error: # then sub_error gets us closer
                if not sub_error <= sum(diffs): # if it is worse
                    params[param_key] += delta  # return parameter to its original value
                # otherwise, sub_error was good, so do noting. See (1).
            else: # then plus_error gets us closer
                if not plus_error <= sum(diffs): #if it is wosse
                    params[param_key] += delta 
                else:
                    # then plus error was good, so return parameter to one delta above its original value.
                    params[param_key] += 2*delta 
            # we loop, and update our polynomials
            numer_func = p.format(**params)
            denom_func = q.format(**params)
        if run == n_runs:
            print("Numerator:  ", numer_func.replace("*", "").replace("k","x")+"\n"\
                  "Denominator:", denom_func.replace("*", "").replace("k","x"))
            break
        run += 1

# Pretty close rational functions:
# 1)  6.900000000000155x^2 + 164.29999999999924x + 47.0 
#     / 95.09999999999914x^5 + 600.3000000000001x^4 + 986.4999999999969x^3 + 800.3000000000001x^2 + 200.29999999999998x + 15.0
# 2)  4.829999999998592x^2 + 166.5099999999859x + 47.00000000007106
#     113.60999999992283x^5 + 567.2550000000298x^4 + 1001.6399999999985x^3 + 800.8349999999992x^2 + 200.83499999999924x + 15.000000000017765


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
                
import math

def cont_fraction(denominators, numerators = None):
    '''
    This computes finite continued fractions of the form:
                        b_0
        a_0 + ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            b_1
              a_1 + ^^^^^^^^^^^^^^^^^^^^^^^
                                b_2
                    a_2 + ^^^^^^^^^^^^^^^^^^
                    ......(and so on until)........
                                    b_n
                            a_n + ^^^^^^^^^^
                                    a_{n+1}
    The a_i are DENOMINATORS 
    The b_i are NUMERATORS
    Therefore, the input are lists 
        denominators = [a_0, ... , a_{n+1}]
        numerators   = [b_0, ... , b_n].
    
    Generally, b_i = 1 for all i. Thus, "numerators" has the 
    default option b_i = 1 for all i, but this option 
    can be overwritten.
    '''   
    if numerators is None:
        numerators = [1]*(len(denominators) - 1)

    assert len(denominators) == len(numerators) + 1, \
    "# of numerators needs to equal number of denominators plus one."
    
    ## Base Case
    if len(denominators) == 1:
        return denominators[0]
    ## Else, we proceed and add 
    a_n = denominators[0] # Grab a_n
    denominators.pop(0)   # Remove it from the list of denominators
    b_n = numerators[0]   
    numerators.pop(0)
    
    return a_n + b_n/(cont_fraction(denominators, numerators))

def find_cfraction(numer, denom, skip = True):
    cden = math.gcd(numer, denom)
    if cden != 1:
        numer = numer/cden
        denom = denom/cden

    # Base case
    if denom == 1:
        return [numer]
    else: 
        a_n = numer//denom
        # print(a_n, numer - denom*a_n)
        return [a_n] + find_cfraction(denom, numer - denom*a_n)


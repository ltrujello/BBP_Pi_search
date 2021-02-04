import math
from polynomials_and_series import series, p_over_q_vals
# We are interested in the image of the multivariable 
# integer function sending coefficients to the power series computed via the coefficients.

def run_tests_on_degree_4():
    for a_0 in [0, 1, 4]:
        for a_1 in [0, 1, 4]:
            for a_2 in [0, 1, 4]:
                for a_3 in [0, 1, 4]:
                    for a_4 in [0, 1, 4]:
                        for b_0 in [1, 4]:
                            for b_1 in [0, 1, 4]:
                                for b_2 in [0, 1, 4]:
                                    for b_3 in [0, 1, 4]:
                                        for b_4 in [0, 1, 4]:
                                            p = str(a_4) + "*x^4 + " + str(a_3) + "*x^3 + " + str(a_2) + "*x^2 + " + str(a_1) + "*x + " + str(a_0)
                                            q = str(b_4) + "*x^4 + " + str(b_3) + "*x^3 + " + str(b_2) + "*x^2 + " + str(b_1) + "*x + " + str(b_0) 
                                            print(\
                                            str(a_0)\
                                            + str(a_1)\
                                            + str(a_2)\
                                            + str(a_3)\
                                            + str(a_4)\
                                            + str(b_0)\
                                            + str(b_1)\
                                            + str(b_2)\
                                            + str(b_3)\
                                            + str(b_4),\
                                            "SUM:",
                                            series(p_over_q_vals(p, q, list(range(0,100)), coeff = 1/10 )))

def run_tests_on_single_a_n():
    for deg in range(0, 20):
        p = "x^" + str(deg)
        q = "1"
        print("SUM: ", series(p_over_q_vals(p, q, list(range(0,200)), coeff = 1/10 )))

def run_tests_on_a_n_in_denom():
    for deg in range(0, 50):
        p = "1" 
        q = "x^" + str(deg)
        print("SUM: ", series(p_over_q_vals(p, q, list(range(1,200)), coeff = 1/10 )))
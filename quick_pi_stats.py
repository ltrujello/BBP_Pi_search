import numpy as np
import math 
from decimal import *
getcontext().prec = 40
'''
Are the digits really uniform? Yes, they are. 
'''
with open("billion_pi.txt", "r") as f:
    pi = f.read()[2:]

def find_pi_digit(digits):
    digits = str(digits)
    n_digits = len(digits)
    i=0
    while i < len(pi):
        if pi[i: i + n_digits] != digits:
            i+=1
        else:
            return "I found " + pi[i: i+n_digits]+ " at the " + str(i) + "th digit of pi."
    return "No match!"
    
def freq_of_digits(start, end):
    window_of_pi = [int(x) for x in list(pi[start:end])]
    pi_np = np.array(window_of_pi)
    return np.unique(pi_np, return_counts=True)[1]

def where_in_pi(start, end, n):
    window_of_pi = [int(x) for x in list(pi[start:end])]
    pi_np = np.array(window_of_pi)
    return np.where(pi_np == n)


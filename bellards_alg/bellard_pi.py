import math 
'''
This is Bellard's algorithm for finding the nth-digit of pi in base 10. It's pretty 
fast compared to other implementations which use BBP. However, it's still not practical 
for computing the largest digit. 
'''

def prod_but_i(k, args):
    '''
    Given a_0 ... a_n, we return 
    a_0*...*a_{k-1}*a_{k+1}*...a_n.
    '''
    prod = 1
    for i,arg in enumerate(args):
        if i != k:
            prod *= arg
    return prod


def mod_one(b, args): 
    '''
    We assume s is of the form s = b/A where 
    A = a_0 * a_1 * ... * a_n 
    where gcd(a_i, a_j) = 1 for i \= j. 
    '''
    b_mods = []
    c_mods = []
    alpha_mods = []
    for i, a_i in enumerate(args):
        b_i = b % a_i
        b_mods.append(b_i)
        
        prod = prod_but_i(i, args)
        c_i = inv_mod(prod, a_i)
        c_mods.append(c_i)

        alpha_mods.append(b_i*c_i % a_i)
    return alpha_mods



def sum_alpha_over_a_i(b,*args):
    sum = 0
    alpha_mods = mod_one(b, args)
    for i, a_i in enumerate(args):
        alpha_i = alpha_mods[i]
        sum += alpha_i/a_i
    return sum



def inv_mod(a, div):
    '''
    Calculates the a^{-1} mod div. 
    This is the number k such that k*a mod div = 1.
    
    We assume a, div are coprime.
    '''

    #### Inverse Modulo Algorithm ####
    # The initial conditions
    v = div
    x = 1 
    y = 0 

    # Loop till we find the divisor
    while a != 0:
        quot = v // a

        t = x # save x for next step
        x = y - quot * x 
        y = t 

        t = a # save a for next step
        a = v - quot * a # a is now the remainder
        v = t 

    return y % div # c might by negative, but python will catch it. Not always true for other programming languages.

def mul_mod(a, b, m):
    return a*b % m

def pow_mod(a, b, m):
    '''
    Calculates a^b mod m. 
    '''
    r = 1
    A = a
    while True:
        # print(A,b)s
        if b & 1:
            r = mul_mod(r,A, m)
        b = b >> 1
        if b == 0:
            break
        A = mul_mod(A,A,m)
    return r


def is_prime(n):
    if (n % 2) == 0:
        return 0

    r = int(math.sqrt(n))
    for i in range(3, r+1,2):
        if (n % i) == 0:
            return 0
    return 1

def next_prime(n):
    n+=1
    while not is_prime(n):
        n+=1
    return n

def nth_digit(n):
    N = int((n + 20) * math.log(10) / math.log(2))
    print(2*N)
    sum = 0
    a = 3
    while a <= 2*N:  
        vmax = int(math.log(2 * N) / math.log(a))
        av = 1
        for i in range(0, vmax):
            av = av * a # maybe here
        s = 0
        num = 1
        den = 1
        v = 0
        kq = 1
        kq2 = 1

        for k in range(1, N+1):
            t = k
            if kq >= a: 
                t = t / a
                v -= 1
                while (t % a) == 0:
                    t = t / a
                    v -= 1
                kq = 0
            kq+=1
            num = mul_mod(num, t, av)
            t = 2*k - 1
            if (kq2 >= a):
                if (kq2 == a):
                    t = t / a
                    v+=1
                    while (t % a) == 0:
                        t = t / a
                        v+=1
                kq2 -= a
            den = mul_mod(den, t, av)
            kq2 += 2

            if v > 0:
                t = inv_mod(den, av)
                t = mul_mod(t, num, av)
                t = mul_mod(t, k, av)
                for i in range(v, vmax):
                    t = mul_mod(t, a, av)
                s += t
                if (s >= av):
                    s -= av

        t = pow_mod(10, n - 1, av)
        s = mul_mod(s, t, av)
        sum = (sum + s/av) % 1.0
        a = next_prime(a)
    print("Decimal digits of pi at position {a}:{b}".format(a = n, b = int(sum*1e9)))

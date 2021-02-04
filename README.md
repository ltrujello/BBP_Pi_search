# What is this
This is an effort to understand the following function:
<img src="https://github.com/ltrujello/Rational_Series/blob/master/main_equation.svg" height="100px" />

(Obviously, sum has issues if the denominator is zero on any of the integers, or if there is no constant term. )
For this function, I am interested in the case for b = 10. It seems that we cannot express certain real numbers 
as the above sum when $b = 10$, but we can however express real numbers when $b = 2$. 
The main question is why. Also, if we can find expressions when $b = 10$, this would lead to very efficient 
n-th digit calculations, but it's unlikely that it will exist. 

The answer of "why" is probably an extremely deep one way beyond my abilities of finding out, 
but I can at least and ask much simpler questions about the behavior of the above function. 

## Questions about the sum:
1. **Is this function defined for all coefficients?** Obviously not, since we can design the denominator to have 
integer roots. What is nice, however, is that if the denominator does not have any obvious singularities, then the sum converge by the ratio 
test (we just need b to be strictly less than one. )

2. **Is the function above a group homomorphism?** No, it is not. If we fix the denominator coefficients and design 
the function to send coefficients to the numerator, then it is. 

3. **If we consider a well-defined set of coefficients, is there a formula for the value of the infinite sum?** 
This is the main, more manageable question I am considering.
For special cases, we can construct a formula. For other cases, there does not seem to be a formula in terms 
of functions we know, but perhaps there is a relationship between the coefficients for which there does not exist 
a nice formula for. 

4. (An interesting difficult question) Suppose we extend the functions to consider all real-value coefficients. 
**How does the the sum change when I perturb the numerator by a small value? How does the sum change when I perturb 
the denominator by a small value? ** Haven't analyzed yet.




import numpy as np
import copy

def findNearestNumbers(x):
    '''
    This function finds the nearest representable real numbers higher (upperNumber) 
    and lower (lowerNumber) than a given floating point value, x.

    The fractional range (the difference between the two numbers divided by x
    is also returned.

    The function expects x to be a numpy float64 value
    '''

    #If the number is negative, we must take note of it but continue
    #as if it were positive.  This works because floating point numbers
    #are symmetric about 0
    flipSign = False
    if x<0:
        flipSign = True
        x = abs(x)

    
    #Find the nearest representable number above x
    if x == 0:
        n = 0
    else:
        n = np.floor(np.log2(x))
    changed = True
    while changed == True:
        #Work in powers of 2, since the computer does too
        result = x + pow(2, n)
        if x == result:
            changed = False #Once result is unchanged, we have found the limit
                            #of what we can represent
        else:
            n-=1
            previousResult = result
    
    #upperNumber is the nearest representable number above x
    upperNumber = copy.copy(previousResult)

    #Find the nearest representable number below x
    if x == 0:
        n = 0
    else:
        n = np.floor(np.log2(x))
    changed = True
    while changed == True:
        result = x - pow(2, n)
        if x == result:
            changed = False
        else:
            n-=1
            previousResult = result

    #lowerNumber is the nearest representable number below x
    lowerNumber = copy.copy(previousResult)


    if flipSign == True:
        #If x was actually negative, then we need to swap the order of the 
        #numbers and change their sign.
        temp = copy.copy(upperNumber)
        upperNumber = copy.copy(lowerNumber)
        lowerNumber = copy.copy(temp)
        x = -x
        upperNumber = -upperNumber
        lowerNumber = -lowerNumber

    #now compute fractional rounding ranges.
    #Must halve the values because they are for rounding.
    if x != 0:
        lowerFractionalRange = 0.5*(lowerNumber - x)/x
        upperFractionalRange = 0.5*(upperNumber - x)/x
    else:
        #Cannot divide by zero, so cannot provide fractional ranges.
        lowerFractionalRange = "Not possible to compute"
        upperFractionalRange = "Not possible to compute"

    return lowerNumber, upperNumber, lowerFractionalRange, upperFractionalRange


#==========================================================================================================
print("")
#First calculate all the values
A = 0.25
C, B, AC_Range, AB_Range = findNearestNumbers(A)
F, E2, CF_Range, CE2_Range = findNearestNumbers(C)
E1, D, BE1_Range, BD_Range = findNearestNumbers(B)

#Some validation... 
if E1 == E2:
    print("Both values for E are identical.")
    print("E = %f" % E1)


#Now print the values in an organised way
print("Answer to 1a:")
print("Lower number (C):  %.100f" % C)
print("Upper number (B):  %.100f" % B)
print("Fractional rounding range (C to A):  %.100f" % AC_Range)
print("Fractioanl rounding range in 2^n form (C to A):  2^%f" % np.log2(abs(AC_Range)))
print("Fractional rounding range (B to A):  %.100f" % AB_Range)
print("Fractioanl rounding range in 2^n form (B to A):  2^%f" % np.log2(AB_Range))

print("")
print("Answer to 1b:")
print("Values:")
print("D:  %.100f" % D)
print("E:  %.100f" % E1)
print("F:  %.100f" % F)

print("Fractional rounding ranges:")
print("F to C:  %.100f" % CF_Range)
print("E to C:  %.100f" % CE2_Range)
print("E to B:  %.100f" % BE1_Range)
print("D to B:  %.100f" % BD_Range)

print("Fractional rounding ranges in 2^n form:")
print("F to C:  2^%f" % np.log2(abs(CF_Range)))
print("E to C:  2^%f" % np.log2(CE2_Range))
print("E to B:  2^%f" % np.log2(abs(BE1_Range)))
print("D to B:  2^%f" % np.log2(BD_Range))

print("")
print("Validation testing:")
print("Test a negative number:")
print(findNearestNumbers(-0.25))
print("Test zero:")
print(findNearestNumbers(0))
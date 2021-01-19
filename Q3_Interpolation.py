import numpy as np
from matplotlib import pyplot as plt
import myMatrixFunctions as m

def lagrangeInterpolation(data, xValues):
    #data is a 2D array of values, of shape (2, n) where n is the number of data points.
    #These will be used to calculate the interpolation.
    #xValues is the array of values that you wish to know the y values of, based on the 
    #interpolation.

    #the order of the polynomial is:
    n = np.shape(data)[1] - 1

    #create array to store the y values that correspond to xValues
    yValues = np.zeros(len(xValues))

    #Implement the equation for Lagrange interpolation given in section 4.4 of the 
    #lecture notes.  This is equation 4.2
    for k in range(len(xValues)):
        total = 0
        for i in range(0, n + 1):
            product = data[1][i]
            for j in range(0, n + 1):
                if i != j:
                    product = product * (xValues[k] - data[0][j])/(data[0][i] - data[0][j])
            total = total + product

        yValues[k] = total    

    return yValues




def cubicSplineInterpolation(data, xValues):
    #data:      x values in zeroth row, y values in 1st row
    #xValues:  The values over which you wish to plot the spline.

    #For the cubic spline, we require more than three points for the function to work.
    if len(data[0]) <= 3:
        print(len(data[0]))
        raise Exception("There is an insufficient number data points to plot a cubic spline")
    #Lengths of x and y in data arrays must match
    if len(data[0]) != len(data[1]):
        raise Exception("The arrays of x and y data must have the same length")


    N = len(xValues)
    n = np.shape(data)[1] - 1

    #First find the second derivatives.  To do this, we implement equation 4.15 in 
    #section 4.5 of the lecture notes as a matrix equation.

    #Assign relevant values to matrices
    M = np.zeros((n - 1, n - 1))
    b = np.zeros((n - 1, 1))
    #The first and last rows of the matrix only have two elements rather than three,
    #so fill them seperately.
    #first row
    M[0][0] = (data[0][2] - data[0][0])/3
    M[0][1] = (data[0][2] - data[0][1])/6
    b[0][0] = ((data[1][2] - data[1][1])/(data[0][2] - data[0][1])) - ((data[1][1] - data[1][0])/(data[0][1] - data[0][0]))
    #last row
    M[n-2][n-3] = (data[0][n-1] - data[0][n-2])/6
    M[n-2][n-2] = (data[0][n] - data[0][n-2])/3
    b[n-2][0] = ((data[1][n] - data[1][n-1])/(data[0][n] - data[0][n-1])) - ((data[1][n-1] - data[1][n-2])/(data[0][n-1] - data[0][n-2]))
    #fill in the rest 
    for i in range(2, n - 1):
        #i refers to the same index as the lecture notes, rather than the matrix index directly.
        M[i-1][i-2] = (data[0][i] - data[0][i-1])/6
        M[i-1][i-1] = (data[0][i+1] - data[0][i-1])/3
        M[i-1][i] = (data[0][i+1] - data[0][i])/6

        b[i-1][0] = ((data[1][i+1] - data[1][i])/(data[0][i+1] - data[0][i])) - ((data[1][i] - data[1][i-1])/(data[0][i] - data[0][i-1]))

    #Now can solve the matrix equation using code written for Q2.
    L, U = m.getDecomposition(M)
    secondDerivatives = m.solveMatrixEquation(L, U, b)
    #Add on the natural spline conditions to the array of second derivatives
    secondDerivatives = np.concatenate(([[0]], secondDerivatives, [[0]]), axis = 0)

    #Create an empty array for the y values to plot.
    yValues = np.zeros(N)

    for i in range(N):
        #First we need to find the indeces 'before' and 'after' of the points in the
        #data array that come before and after our x value of interest.
        found = False
        j = 0
        while (found == False)and (j <= n):
            if xValues[i] >= data[0][j]:
                j += 1
            else:
                found = True
        
        before = int(j - 1)
        #We may get 'before' to be the last element, which is incorrect.
        #This can be corrected by subtracting 1 from it.
        if before == int(n):
            before = before - 1
        #'after' is always the next index
        after = before + 1

        #Now the coefficients can be calculated
        #Use equations 4.5, 4.6, 4.8 and 4.9 from section 4.5 in the notes
        A = (data[0][after] - xValues[i])/(data[0][after] - data[0][before])
        B = (xValues[i] - data[0][before])/(data[0][after] - data[0][before])

        C = ((pow(A, 3) - A)*pow((data[0][after] - data[0][before]), 2))/6
        D = ((pow(B, 3) - B)*pow((data[0][after] - data[0][before]), 2))/6

        #The relevant y-value can now be found.  This is equation 4.7 in the lecture notes
        yValues[i] = A*data[1][before] + B*data[1][after] + C*secondDerivatives[before][0] + D*secondDerivatives[after][0]

    return yValues

    

#load in the given data
data = np.array([[-0.75, -0.5, -0.35, -0.1, 0.05, 0.1, 0.23, 0.29, 0.48, 0.6, 0.92, 1.05, 1.5], 
[0.10, 0.30, 0.47, 0.66, 0.60, 0.54, 0.30, 0.15, -0.32, -0.54, -0.60, -0.47, -0.08]])

#we wish to find function values at the following points:
xValues = np.linspace(data[0][0], data[0][np.shape(data)[1] - 1], 1000)
# xValues = np.linspace(data[0][0], data[0][np.shape(data)[1] - 1], 20)

yValuesLagrange = lagrangeInterpolation(data, xValues)
yValuesCubic = cubicSplineInterpolation(data, xValues)



#Answer to part c:
plt.figure(1)
plt.plot(data[0], data[1], marker = "x", color = "black", linestyle = "none", label = "Data points")
plt.plot(xValues, yValuesLagrange, color = "red", label = "Lagrange interpolation")
plt.plot(xValues, yValuesCubic, color = "blue", label = "Cubic spline interpolation")
plt.title("Interpolation comparison")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("figures/interpolationComparison.eps", format = "eps", dpi = 1000)

#also show cubic spline on its own, since it's hard to see with the lagrange one there too.
plt.figure(2)
plt.plot(data[0], data[1], marker = "x", color = "black", linestyle = "none", label = "Data points")
plt.plot(xValues, yValuesCubic, color = "blue", label = "Cubic spline interpolation")
plt.title("Cubic spline interpolation only")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("figures/interpolationComparison2.eps", format = "eps", dpi = 1000)


plt.show()
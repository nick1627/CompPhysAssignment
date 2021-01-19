import numpy as np
from matplotlib import pyplot as plt



def RHS_func(t, V_out):
    #The function f in dx/dt = f(x, t)
    #This will be provided to solveDifferentialEquation()
    V0 = 1
    if t<0:
        V_in = V0
    else:
        V_in = 0

    return V_in/V0 - V_out

def trueSolution(t):
    #The analytical solution to the equation, which will be used for 
    #checking the errors.
    return np.exp(-t)

def RHS_func_2(t, V_out, T):
    #Another different function for the right hand side of the differential
    #equation.

    V0 = 1

    #create square wave
    if t%T < T/2:
        V_in = 0
    else:
        V_in = V0
    
    return V_in/V0 - V_out
    


def solveDifferentialEquation(f, x0, tValues, method, *funcArgs):
    #f: the function on the right hand side of the differential equation when 
    # it is of the form dx/dt = f(x, t)
    #x0:  the initial value for x at the first time value.
    #tValues:  The time values to solve over
    #method signifies which method to use
        #method = 1 means use 4th order Runge Kutta method
        #method = 2 means use 4th order Adams-Bashforth method
    #*funcArgs:  Any additional arguments required for f.
    
    h = tValues[1] - tValues[0]

    if method == 1:
        #Runge-Kutta
        xValues = np.array([x0])
        for i in range(len(tValues)-1):
            #Implement equations 8.47 a-e in the lecture notes.
            fa = f(tValues[i], xValues[i], *funcArgs)
            fb = f(tValues[i] + h/2, xValues[i] + h*fa/2, *funcArgs)
            fc = f(tValues[i] + h/2, xValues[i] + h*fb/2, *funcArgs)
            fd = f(tValues[i] + h, xValues[i] + h*fc, *funcArgs)
            
            xValues = np.append(xValues, xValues[-1] + (h/6)*(fa + 2*fb + 2*fc + fd))

    else:
        #Adams-Bashforth
        #Since Adams-Bashforth is a multi-step method, we need to get some 
        #starting values.  It's fourth order, so we can only start iterating
        #at index i = 3.
        #We therefore need values with index 0, 1, 2 and 3 from a different method.
        #We will reuse the Runge-Kutta method given above.

        start_tValues = np.array([tValues[0], tValues[1], tValues[2], tValues[3]])
        xValues = solveDifferentialEquation(f, x0, start_tValues, 1, *funcArgs)

        #Now that the start values have been found, we can continue with the AB method
        #as described in the lecture notes.
        for i in range(3, len(tValues)-1):
            #Implement equations 8.24 and 8.25 in the lecture notes
            fa = f(tValues[i], xValues[i], *funcArgs)
            fb = f(tValues[i-1], xValues[i-1], *funcArgs)
            fc = f(tValues[i-2], xValues[i-2], *funcArgs)
            fd = f(tValues[i-3], xValues[i-3], *funcArgs)

            xValues = np.append(xValues, xValues[-1] + (h/24)*(55*fa - 59*fb + 37*fc - 9*fd))

    return xValues


#Part c

#create some time values
spacing = pow(10, -3)
tValues_c = np.arange(0, 10 + spacing, spacing)

#Solve the DE two ways, and also get the analytical solution
V_out_RK = solveDifferentialEquation(RHS_func, 1, tValues_c, 1)
V_out_AB = solveDifferentialEquation(RHS_func, 1, tValues_c, 2)
V_out_true = trueSolution(tValues_c)

#Calculate relative residuals using the analytical solution
relativeResiduals_RK = (V_out_RK - V_out_true)/V_out_true
relativeResiduals_AB = (V_out_AB - V_out_true)/V_out_true

#Find gradients of residuals by looking at line endpoints
resGrad_RK = (relativeResiduals_RK[-1] - relativeResiduals_RK[0])/(tValues_c[-1] - tValues_c[0])
resGrad_AB = (relativeResiduals_AB[-1] - relativeResiduals_AB[0])/(tValues_c[-1] - tValues_c[0])

plt.figure(1)
plt.plot(tValues_c, V_out_RK, label = "Runge-Kutta", color = "red")
plt.plot(tValues_c, V_out_AB, label = "Adams-Bashforth", color = "blue")
plt.plot(tValues_c, V_out_true, label = "Analytical solution", color = "black")
plt.legend()
plt.title("V_out for part c")
plt.xlabel("t/CR")
plt.ylabel("V_out/V0")
plt.savefig("figures/diffEqSolnC.eps", type = "eps", dpi = 1000)

#Now take a look at the relative residuals

plt.figure(2)
plt.plot(tValues_c, relativeResiduals_RK, label = "Runge-Kutta", color = "red")
plt.plot(tValues_c, relativeResiduals_AB, label = "Adams-Bashforth", color = "blue")
plt.legend()
plt.title("Relative residuals")
plt.ylabel("Relative residuals")
plt.xlabel("t/CR")
plt.savefig("figures/cResiduals.eps", format = "eps", dpi = 1000)

#print the values of the gradients of the relative residuals
print("Part c")
print("Printing the gradients of the relative residuals")
print("Runge-Kutta:  %f x10^-11" % (resGrad_RK*10**11))
print("Adams-Bashforth:  %f x10^-10" % (resGrad_AB*10**10))
print("Gradient of AB divided by gradient of RK:  %f" % (resGrad_AB/resGrad_RK))


#part d
#Need different times because of the different periods
tValues_d1 = np.arange(0, 10 + 2*spacing, 2*spacing)
tValues_d2 = np.arange(0, 10 + spacing/2, spacing/2)

#solve DE again, both Runge-Kutta this time
V_out_RK_d1 = solveDifferentialEquation(RHS_func, 1, tValues_d1, 1)
V_out_RK_d2 = solveDifferentialEquation(RHS_func, 1, tValues_d2, 1)
V_out_true_d1 = trueSolution(tValues_d1)
V_out_true_d2 = trueSolution(tValues_d2)

#Find relative residuals
relativeResiduals_d1 = (V_out_RK_d1 - V_out_true_d1)/V_out_true_d1
relativeResiduals_d2 = (V_out_RK_d2 - V_out_true_d2)/V_out_true_d2

#Plot relative residuals
plt.figure(3)
plt.plot(tValues_d1, relativeResiduals_d1, label = "Doubled step size")
plt.plot(tValues_d2, relativeResiduals_d2, label = "Halved step size")
plt.plot(tValues_c, relativeResiduals_RK, label = "Normal step size")
plt.xlabel("t/CR")
plt.ylabel("Relative residuals")
plt.legend()
plt.savefig("figures/dResiduals.eps", format = "eps", dpi = 1000)

#Find the gradients of the residuals
resGrad_d1 = (relativeResiduals_d1[-1] - relativeResiduals_d1[0])/(tValues_d1[-1] - tValues_d1[0])
resGrad_d2 = (relativeResiduals_d2[-1] - relativeResiduals_d2[0])/(tValues_d2[-1] - tValues_d2[0])


print("Part d")
print("Printing the gradients of the relative residuals")
print("Doubled step size:  %f" % (resGrad_d1))
print("Normal step size:  %f" % (resGrad_d2))
print("Halved step size: %f" % (resGrad_d2))
print("Gradient for doubled step size divided by gradient for normal step:  %f" % (resGrad_d1/resGrad_RK))
print("Gradient for normal step size divided by gradient for halved step:  %f" % (resGrad_RK/resGrad_d2))


#part e
#Halve and double the period, and solve again
period = 1/2
V_out_RK_square_short = solveDifferentialEquation(RHS_func_2, 1, tValues_c, 1, period)
period = 2
V_out_RK_square_long = solveDifferentialEquation(RHS_func_2, 1, tValues_c, 1, period)

#Plot the results...
plt.figure(4)
plt.plot(tValues_c, V_out_RK_square_short, label = "T = RC/2", color = "red")
plt.plot(tValues_c, V_out_RK_square_long, label = "T = 2RC", color = "blue")
plt.xlabel("t/CR")
plt.ylabel("V_out")
plt.title("Comparison of results with different periods")
plt.legend()
plt.savefig("figures/diffEqSolnE.eps")


plt.show()
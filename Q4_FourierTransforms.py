import numpy as np
import matplotlib.pyplot as plt

def signalFunc(t):
    #The signal function as defined in the question
    if (t>=5) and (t<=7):
        return 4
    else:
        return 0
    
def responseFunc(t):
    #The gaussian response function as defined in the question
    return (1/np.sqrt(2*np.pi))*np.exp(-pow(t, 2)/4)

#we wish to convolve signalFunc and responseFunc

#First let's plot the functions
#m was chosen to reduce aliasing.  See the write-up for a detailed explanation.
m = 12 
N = pow(2, m) #N is the number of samples

tValues = np.linspace(-50, 50, N)

#define the step size:
spacing = tValues[1] - tValues[0]

signal_y = np.zeros(len(tValues))
for i in range(len(tValues)):
    signal_y[i] = signalFunc(tValues[i])

response_y = responseFunc(tValues)

#Plot the functions
plt.figure(1)
plt.plot(tValues, signal_y, color = "blue", label = "y = h(t)")
plt.plot(tValues, response_y, color = "red", label = "y = g(t)")
plt.legend()
plt.xlabel("t")
plt.ylabel("y")
plt.title("Signals to be convolved")
plt.savefig("figures/originalFunctions.eps", format = "eps", dpi = 1000)

#Now find the fourier transforms
#we will sample both functions in the same places
F_t = np.fft.fftfreq(len(tValues), spacing)
#Account for 2pi scaling
F_t = 2*np.pi*F_t
#Get Fourier transforms of functions
F_signal = np.fft.fft(signal_y)
F_response = np.fft.fft(response_y)


#Plot the Fourier transforms
plt.figure(2)
plt.plot(F_t, F_signal, color = "blue", label = "F[h(t)]")
plt.plot(F_t, F_response, color = "red", label = "F[g(t)]")
plt.legend()
plt.title("Fourier transformed signals")
plt.savefig("figures/transformedFunctions.eps", format = "eps", dpi = 1000)

#Multiply the two fourier transforms together to get the transformed convolution
F_convolution = F_signal*F_response

#Perform inverse fourier transform
convolution = np.fft.ifft(F_convolution)
convolution = np.fft.ifftshift(convolution)
#Account for scaling as detailed in the lecture notes.
convolution = convolution*spacing

#Plot the convolved function
plt.figure(3)
plt.plot(tValues, convolution)
plt.title("Convolution")
plt.xlabel("Time")
plt.savefig("figures/convolvedFunctions.eps", format = "eps", dpi = 1000)



plt.show()


import numpy as np
import scipy as sp
import myMatrixFunctions as m

print("")
#a)     See function myMatrixFunctions.LU_decomposition()

#b)     See below:  
print("Answer to part b:")
A = np.array([[3, 1, 0, 0, 0], [3, 9, 4, 0, 0], [0, 8, 20, 10, 0], [0, 0, -22, 31, -25], [0, 0, 0, -35, 61]])
L, U = m.getDecomposition(A)
print("L:  ")
print(L)
print("U:  ")
print(U)
print("LU")
print(m.matMul(L, U))
print("")

#Since A = LU, det(A) = det(L)*det(U)
#For lower and upper diagonal matrices, the determinant is simply the product of the diagonal elements.
#For this decomposition method, the product of the diagonals of L is always 1.
#So det(A) is simply the product of the diagonals of U:
detA = 1
for i in range(np.shape(U)[0]):
    detA = detA*U[i][i]
print("The determinant of A is %.100f" % detA)
print("Using numpy, the determinant is:  ")
print(np.linalg.det(A))
print("")


#c)     see function solveMatrixEquation()

#d)
print("Answer to part d")
b = np.array([[2], [5], [-4], [8], [9]])
x = m.solveMatrixEquation(L, U, b)
print("x = ")
print(x)

print(m.matMul(A, x))
print("")
#e)
print("Answer to part e")
#We want to solve A A^-1 = I for A^-1
#We can split the identity up into columns and solve for 
#each column in A^-1 using the code already written.
#define an empty inverse matrix
inverseA = np.zeros(np.shape(A))
for j in range(np.shape(A)[1]):
    #Create the jth column of the identity
    identityColumn = np.zeros((np.shape(A)[0], 1))
    identityColumn[j][0] = 1

    #solve for the jth column in the inverse matrix
    inverseColumn = m.solveMatrixEquation(L, U, identityColumn)
    #assign the column to the correct position in the inverse matrix
    inverseA[:,j] = inverseColumn[:,0]

print("The inverse of A is:")
print(inverseA)
#Check result by finding the identity
print("A on inverse A is:")
identity = m.matMul(A, inverseA)
print(identity)
#Round for clarity
print("The rounded A on A^-1 is")
print(np.round(identity, decimals = 4))



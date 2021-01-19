import numpy as np
import copy

#Note that the matrix functions given in this file assume that vectors are
#defined vertically, i.e some vector v is np.array([[1], [2], [3]]) rather
#than np.array([1, 2, 3]).  This was done for generality, since one may 
#actually want to use a 1x3 matrix instead of a 3x1 in some circumstances,
#even though using 3x1 vectors is far more common.


def LU_decomposition(M):
    #M is a square input matrix

    rows, cols = np.shape(M)

    L = np.zeros(np.shape(M))
    U = np.zeros(np.shape(M))

    # In the following set of loops, we could have performed the operations
    # on a matrix that combined L and U to save memory space.  However, the 
    # decision was made to prioritise speed, since combining them would have
    # meant using extra 'if' statements to check the value of the diagonal.
    # These if statements would have had to run at least once for each
    # element, making the whole process more time consuming for larger
    # matrices.


    #Implement method for LU decomposition described in section 3.4 of the lecture notes.
    for j in range(0, cols):
        L[j][j] = 1  #use Doolittle choice
        for i in range(0, j + 1):
            total = 0
            for k in range(0, i):
                total = total + L[i][k] * U[k][j]

            U[i][j] = M[i][j] - total
    
        for i in range(j + 1, rows):
            total = 0
            for k in range(0, j):
                total = total + L[i][k] * U[k][j]
            
            L[i][j] = (1/U[j][j]) * (M[i][j] - total)

    #Copy the elements into a single matrix for returning.
    R = copy.deepcopy(U)
    for i in range(1, rows):
        for j in range(0, i):
            R[i][j] = copy.copy(L[i][j])


    #Return R, the single matrix that combines L and U.
    #L and U can be recovered by separating along the diagonal
    #and using L[i][i] = 1.
    return R


def getDecomposition(M):
    # This function performs LU decomposition on M using LU_decomposition(), but
    # puts the output in the form L U.

    R = LU_decomposition(M)

    #Now extract L and U from R.  All values can be copied out except the diagonal
    #elements of L, which are all 1.
    shape = np.shape(M)
    L = np.zeros(shape)
    U = np.zeros(shape)

    #Loop through R, copying out the elements.
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            if i == j:
                L[i][j] = 1
                U[i][j] = R[i][j]
            elif i < j:
                U[i][j] = R[i][j]
            else:
                L[i][j] = R[i][j]

    return L, U
    

def solveMatrixEquation(L, U, b):
    #This function solves the matrix equation Ax = b for x, where
    #A = LU.

    n = np.shape(b)[0]
    x = np.zeros(np.shape(b))
    y = np.zeros(np.shape(b))

    #The following algorithms are based on the methods for forward
    #and bakward substitution described in section 3.4 of the lecture
    #notes. 
    
    #First need to solve Ly = b using forward substitution
    y[0][0] = b[0][0]/L[0][0]
    
    for i in range(1, n):
        total = 0
        for j in range(0, i):
            total += L[i][j]*y[j][0]

        y[i][0] = (1/L[i][i])*(b[i][0] - total)

    #Need to solve Ux = y using backward substitution
    x[n-1][0] = y[n-1][0]/U[n-1][n-1]

    for i in range(n-2, -1, -1):
        total = 0
        for j in range(i+1, n):
            total += U[i][j]*x[j][0]
        
        x[i][0] = (1/U[i][i])*(y[i][0] - total)
    

    return x


def matMul(A, B):
    #This function performs matrix multiplication.

    #requires input matrices A and B
    #These must both be 2D
    #computes product C = AB

    A_height = np.shape(A)[0]    
    A_width = np.shape(A)[1]
    B_height = np.shape(B)[0]
    B_width = np.shape(B)[1]

    if A_width == B_height:
        #do matrix multiplication
        result = np.zeros((A_height, B_width))
        for i in range(0, A_height):
            for j in range(0, B_width):
                for k in range(0, A_width):
                    result[i][j] = result[i][j] + A[i][k]*B[k][j]

    else:
        #cannot perform matrix multiplication if
        #the number columns in A does not match the 
        #number of rows in B.
        result = "Illegal operation"

    return result


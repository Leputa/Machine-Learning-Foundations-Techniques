from cvxopt import solvers,matrix
import numpy as np

X=np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y=np.array([-1,-1,-1,+1,+1,+1,+1])
Q=np.zeros((7,7))
def kernelfunc(x1,x2):
	return (1+np.dot(x1,x2))**2
for i in range(7):
	for j in range(7):
		Q[i][j]=y[i]*y[j]*kernelfunc(X[i],X[j])
q=matrix(-np.ones((7)))
P=matrix(Q)
A=np.zeros((9,7))
for i in range(2,9):
	A[i][i-2]=-1
A[0,:]=y
A[1,:]=-y
G=matrix(A)
h=matrix(np.zeros(9))

sol=solvers.qp(P,q,G,h)
a=np.array(sol['x']).flatten()
print("**************************************************************************")
print("第2题答案如下：")
print ('Sum of a:'+str(np.sum(a)))
print ('最大索引:'+str(np.argmax(a)+1))
print('最小索引:'+str(np.argmin(a)+1))
print("**************************************************************************")

import numpy as np

class Neural_Network():

	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,3))
		index=0
		for line in lines:                                                                                                       
			dataSet[index]=line.strip().split(' ')
			index+=1
		return dataSet

	def tanh(self,s):
		return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))

	def tanhOfDeriv(self,s):
		return 1-self.tanh(s)**2

	def inference(self,X,Y,W,M):
		m=np.shape(X[0])[0]
		S=[0]*3
		S[1]=np.dot(X[0],W[0])
		X[1][:,1:]=self.tanh(S[1])
		S[2]=np.dot(X[1],W[1]).flatten()
		X[2]=np.sum((Y-S[2])**2)/m
		return X,S

	def backprop(self,X,Y,W,S,eta):
		m=np.shape(X[0])[0]
		Theta=[0]*3
		Theta[2]=2*(S[2]-Y)
		Theta[1]=np.array(np.mat(Theta[2]).T*np.mat(W[1][1:,:]).T)*self.tanhOfDeriv(S[1])
		W[0]=W[0]-eta*np.array(np.mat(X[0]).T*np.mat(Theta[1]))/m
		W[1]=W[1]-eta*np.array(np.mat(X[1]).T*np.mat(Theta[2]).T)/m
		return W

	def train(self,dataSet,M,r,eta,T=50000):
		Y=dataSet[:,-1]
		X,W=self.construct(dataSet,M,r)
		for i in range(T):
			X,S=self.inference(X,Y,W,M)
			W=self.backprop(X,Y,W,S,eta)
		return W

	def validation(self,dataSet,W,M):
		m,n=np.shape(dataSet)
		Y=dataSet[:,-1]
		X=[0]*3
		x0=dataSet[:,:-1].copy()
		one=np.ones(m)
		X[0]=np.insert(x0,0,values=one,axis=1)
		X[1]=np.ones((m,M+1))		
		X,S=self.inference(X,Y,W,M)
		return np.sum(np.sign(S[2])!=Y)/m

	def construct(self,dataSet,M,r):
		m,n=np.shape(dataSet)
		X=[0]*3
		W=[0]*2
		x0=dataSet[:,:-1].copy()
		one=np.ones(m)
		X[0]=np.insert(x0,0,values=one,axis=1)
		X[1]=np.ones((m,M+1))
		W[0]=np.random.uniform(-r,r,size=[n,M])
		W[1]=np.random.uniform(-r,r,size=[M+1,1])
		return X,W

def getBestM(trainingSet,testingSet,nn):
	r=0.1
	eta=0.1
	M=[1,6,11,16,21]
	minEout=100
	bestM=0
	for m in M:
		W=nn.train(trainingSet,m,r,eta,50000)	
		Eout=nn.validation(testingSet,W,m)
		print("M="+str(m)+" and "+"the 0/1 error is:"+str(Eout))
		if Eout<minEout:
			minEout=Eout
			bestM=m
	print()
	print("**************************************************************************")
	print("第11题答案如下：")
	print ('M:'+str(bestM)+' results in the lowest 0/1 Eout:')
	print("**************************************************************************")
	print()


def getBestR(trainingSet,testingSet,nn):
	M=3
	eta=0.1
	R=[0,0.001,0.1,10,1000]
	minEout=100
	bestR=0
	for r in R:
		W=nn.train(trainingSet,M,r,eta,50000)	
		Eout=nn.validation(testingSet,W,M)
		print("r="+str(r)+" and "+"the 0/1 error is:"+str(Eout))
		if Eout<minEout:
			minEout=Eout
			bestR=r
	print()
	print("**************************************************************************")
	print("第12题答案如下：")
	print ('r:'+str(bestR)+' results in the lowest 0/1 Eout:')
	print("**************************************************************************")
	print()


def getBestEta(trainingSet,testingSet,nn):
	M=3
	r=0.1
	Eta=[0.001,0.01,0.1,1,10]
	minEout=100
	bestEta=0
	for eta in Eta:
		W=nn.train(trainingSet,M,r,eta,50000)	
		Eout=nn.validation(testingSet,W,M)
		print("eta="+str(eta)+" and "+"the 0/1 error is:"+str(Eout))
		if Eout<minEout:
			minEout=Eout
			bestEta=eta
	print()
	print("**************************************************************************")
	print("第13题答案如下：")
	print ('eta:'+str(bestEta)+' results in the lowest 0/1 Eout')
	print("**************************************************************************")
	print()


def main():
	nn=Neural_Network()
	trainingSet=nn.file2matrix('hw4_nnet_train.dat')
	testingSet=nn.file2matrix('hw4_nnet_test.dat')
	getBestM(trainingSet,testingSet,nn)
	getBestR(trainingSet,testingSet,nn)
	getBestEta(trainingSet,testingSet,nn)
	



if __name__=='__main__':
	main()










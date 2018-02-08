#没有进行500次重复实验（耗时较多，实验结果有一定随机性）
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

	def inference(self,X,Y,W,level):
		m=np.shape(X[0])[0]
		S=[0]*(level+2)
		for i in range(1,level+2):
			if i==level+1:   #最后一层
				S[i]=np.dot(X[i-1],W[i-1])
				X[i]=np.sum((Y-S[i])**2)/m
			else:
				S[i]=np.dot(X[i-1],W[i-1])
				X[i][:,1:]=self.tanh(S[i])
		return X,S

	def backprop(self,X,Y,W,S,level,eta):
		m=np.shape(X[0])[0]
		Theta=[0]*(level+2)
		Theta[level+1]=2*(S[level+1]-Y)
		for i in range(level,0,-1):
			Theta[i]=np.array(np.mat(Theta[i+1])*np.mat(W[i][1:,:]).T)*self.tanhOfDeriv(S[i])
		for i in range(level+1):
			W[i]-=eta*np.array(np.mat(X[i]).T*np.mat(Theta[i+1]))/m
		return W

	def train(self,dataSet,level,M,r,eta,T=50000): #level 隐藏层层数 
		Y=dataSet[:,-1]
		Y=np.array(np.mat(Y).T)
		X,W=self.construct(dataSet,level,M,r)
		for i in range(T):
			X,S=self.inference(X,Y,W,level)
			W=self.backprop(X,Y,W,S,level,eta)
		return W

	def validation(self,dataSet,W,level,M):
		m,n=np.shape(dataSet)
		Y=dataSet[:,-1]
		Y=np.array(np.mat(Y).T)
		X=self.construct(dataSet,level,M,1)[0]	
		X,S=self.inference(X,Y,W,level)
		return np.sum(np.sign(S[level+1])!=Y)/m

	def construct(self,dataSet,level,M,r):
		m,n=np.shape(dataSet)
		X=[0]*(level+2)
		W=[0]*(level+1)
		x0=dataSet[:,:-1].copy()
		one=np.ones(m)
		for i in range(level+1):
			if i==0:
				X[i]=np.insert(x0,0,values=one,axis=1)
				W[i]=np.random.uniform(-r,r,size=[n,M[i]])
			elif i==level:
				X[i]=np.ones((m,M[i-1]+1))	
				W[i]=np.random.uniform(-r,r,size=[M[i-1]+1,1])
			else:
				X[i]=np.ones((m,M[i-1]+1))
				W[i]=np.random.uniform(-r,r,size=[M[i-1]+1,M[i]])
		return X,W

def getBestM(trainingSet,testingSet,nn):
	level=1
	r=0.1
	eta=0.1
	M=[[1],[6],[11],[16],[21]]
	minEout=100
	bestM=0
	for m in M:
		W=nn.train(trainingSet,level,m,r,eta)	
		Eout=nn.validation(testingSet,W,level,m)
		print("M="+str(m)+" and "+"the 0/1 error is:"+str(Eout))
		if Eout<minEout:
			minEout=Eout
			bestM=m
	print()
	print("**************************************************************************")
	print("第11题答案如下：")
	print ('M:'+str(bestM[0])+' results in the lowest 0/1 Eout:')
	print("**************************************************************************")
	print()


def getBestR(trainingSet,testingSet,nn):
	M=[3]
	level=1
	eta=0.1
	R=[0,0.001,0.1,10,1000]
	minEout=100
	bestR=0
	for r in R:
		W=nn.train(trainingSet,level,M,r,eta)	
		Eout=nn.validation(testingSet,W,level,M)
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
	M=[3]
	level=1
	r=0.1
	Eta=[0.001,0.01,0.1,1,10]
	minEout=100
	bestEta=0
	for eta in Eta:
		W=nn.train(trainingSet,level,M,r,eta)	
		Eout=nn.validation(testingSet,W,level,M)
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

	W=nn.train(trainingSet,2,[8,3],0.1,0.01)
	Eout=nn.validation(testingSet,W,2,[8,3])
	print()
	print("**************************************************************************")
	print("第14题答案如下：")
	print ('Eout with d-8-3-1 neural network is:'+str(Eout))
	print("**************************************************************************")
	print()	

if __name__=='__main__':
	main()










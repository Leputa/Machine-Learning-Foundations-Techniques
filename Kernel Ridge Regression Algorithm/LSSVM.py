import numpy as np

class LSSVM:
	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		trainingSet=np.zeros((400,11))
		testingSet=np.zeros((m-400,11))
		index=0
		for line in lines:
			line=line.strip()
			listFromLine=line.split(' ')
			if(index<400):
				trainingSet[index]=listFromLine
			else:
				testingSet[index-400]=listFromLine
			index+=1
		trainOne=np.ones(400)
		testOne=np.ones(m-400)
		trainingSet=np.insert(trainingSet, 0, values=trainOne, axis=1)
		testingSet=np.insert(testingSet, 0, values=testOne, axis=1)
		return trainingSet,testingSet

	def RBFkernel(self,dataSet,gamma):
		m=len(dataSet)
		K=np.mat(np.zeros((m,m)))
		X=np.mat(dataSet[:,:-1])
		for i in range(m):
			K[:,i]=self.kernerTrans(X,X[i,:],gamma)
		return K

	def kernerTrans(self,X,x,gamma):
		m,n=np.shape(X)
		K=np.mat(np.zeros((m,1)))
		for j in range(m):
			deltaRow=X[j,:]-x
			K[j]=deltaRow*deltaRow.T
		K=np.exp(-gamma*K)
		return K

	def ridgeRegression(self,K,trainingSet,lamda):
		m,n=np.shape(trainingSet)
		y=np.mat(trainingSet[:,-1:])
		belta=(np.eye(m)*lamda+K).I*y
		return belta

	def getError(self,belta,trainingSet,dataSet,gamma):
		m=len(dataSet)
		trainingSet=np.mat(trainingSet[:,:-1])
		y=dataSet[:,-1:].flatten()
		dataSet=np.mat(dataSet[:,:-1])
		error=0
		for i in range(m):
			kernelEval=self.kernerTrans(trainingSet,dataSet[i,:],gamma)
			if np.sign(kernelEval.T*belta)!=y[i]:
				error+=1
		return error/m

	def testError(self,trainingSet,testingSet):
		lamdaList=[0.001,1,1000]
		gammaList=[32,2,0.125]
		minEin=9999999
		minEout=999999
		for i in gammaList:
			Kin=self.RBFkernel(trainingSet,i)
			for j in lamdaList:
				belta=np.mat(self.ridgeRegression(Kin,trainingSet,j))
				Ein=self.getError(belta,trainingSet,trainingSet,i)
				Eout=self.getError(belta,trainingSet,testingSet,i)
				if Ein<minEin:
					minEin=Ein
				if Eout<minEout:
					minEout=Eout
		print("**************************************************************************")
		print("第19题答案如下：")
		print ('Ein:'+str(minEin))
		print("**************************************************************************")
		print()
		print("**************************************************************************")
		print("第20题答案如下：")
		print ('Eout:'+str(minEout))
		print("**************************************************************************")
		print()

def main():
	lssvm=LSSVM()
	trainingSet,testingSet=lssvm.file2matrix('hw2_lssvm_all.dat')
	lssvm.testError(trainingSet,testingSet)

if __name__=="__main__":
	main()
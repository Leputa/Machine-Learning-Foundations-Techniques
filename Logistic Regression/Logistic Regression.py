import numpy as np
import random


class Logistic_Regression():
	def __init__(self):
		self.a=0.001
		self.T=2000
		self.w=np.zeros(21)

	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,21))
		index=0
		for line in lines:
			line=line.strip()
			listFromLine=line.split('\t')
			numList=listFromLine[0].split(' ')                                                                                                         
			dataSet[index]=numList
			index+=1
		one=np.ones(m)
		dataSet=np.insert(dataSet, 0, values=one, axis=1)
		return dataSet

	def getGradient(self,trainingSet):
		m=len(trainingSet)
		X=trainingSet[:,0:21]
		Y=trainingSet[:,21]
		X=np.mat(X)   #m*21
		Y=np.mat(Y).T #m*1
		return np.array(-1/m*(np.multiply(X,Y)).T*(1/(1+np.exp(np.multiply(Y,X*np.mat(self.w).T))))).reshape(21)

	def stochasticGradientDescent(self,trainingSet,a):
		n=1
		m=len(trainingSet)
		for i in range(self.T):
			self.w=self.w-a*self.getGradient(trainingSet[:n,:])
			n=(n+1)%m
			if n==0:
				n=1


	def Iteration(self,trainingSet,a):
		for i in range(self.T):
			self.w=self.w-a*self.getGradient(trainingSet)

	def get0_1Error(self,testingSet):
		m=len(testingSet)
		error=0
		for data in testingSet:
			if np.dot(self.w,data[:21])*data[-1]<=0:
				error+=1
		return error/m

def main():
	logistic_gegression=Logistic_Regression()
	trainingSet=logistic_gegression.file2matrix('hw3_train.dat')
	logistic_gegression.Iteration(trainingSet,0.001)
	testingSet=logistic_gegression.file2matrix('hw3_test.dat')
	error0_1=logistic_gegression.get0_1Error(testingSet)
	print("**************************************************************************")
	print("第18题答案如下：")
	print ('0/1错误率:'+str(error0_1))
	print("**************************************************************************")
	print()

	logistic_gegression=Logistic_Regression()
	trainingSet=logistic_gegression.file2matrix('hw3_train.dat')
	logistic_gegression.Iteration(trainingSet,0.01)
	testingSet=logistic_gegression.file2matrix('hw3_test.dat')
	error0_1=logistic_gegression.get0_1Error(testingSet)
	print("**************************************************************************")
	print("第19题答案如下：")
	print ('0/1错误率:'+str(error0_1))
	print("**************************************************************************")
	print()

	logistic_gegression=Logistic_Regression()
	trainingSet=logistic_gegression.file2matrix('hw3_train.dat')
	logistic_gegression.stochasticGradientDescent(trainingSet,0.001)
	testingSet=logistic_gegression.file2matrix('hw3_test.dat')
	error0_1=logistic_gegression.get0_1Error(testingSet)
	print("**************************************************************************")
	print("第20题答案如下：")
	print ('0/1错误率:'+str(error0_1))
	print("**************************************************************************")
	print()

if __name__=="__main__":
	main()




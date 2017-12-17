import numpy as np
import math

class Regularized_Linear_Regression():
	def __init__(self):
		self.w=np.zeros(3)

	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,3))
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

	def iteration(self,trainingSet,lamb):
		m=len(trainingSet)
		X=np.mat(trainingSet[:,:3])       #m*3
		Y=np.mat(trainingSet[:,3]).T      #m*1
		I=np.mat(np.eye(3))
		self.w=np.array((X.T*X+lamb*I).I*(X.T*Y)).reshape(3)

	def getEin(self,trainingSet,lamb):
		m=len(trainingSet)
		X=np.mat(trainingSet[:,:3])       #m*3
		Y=np.mat(trainingSet[:,3]).T      #m*1
		W=np.mat(self.w).T                #3*1
		return (1/m*((X*W-Y).T*(X*W-Y)+lamb*W[1:,:].T*W[1:,:]))[0,0]

	def getEout(self,testingSet):
		m=len(testingSet)
		X=np.mat(testingSet[:,:3])       #m*3
		Y=np.mat(testingSet[:,3]).T      #m*1
		W=np.mat(self.w).T               #3*1
		return (1/m*((X*W-Y).T*(X*W-Y)))[0,0]

	def get0_1Error(self,dataSet):
		m=len(dataSet)
		error=0
		for data in dataSet:
			if np.dot(self.w,data[:3])*data[-1]<=0:
				error+=1
		return error/m


def main():
	ridge_regression=Regularized_Linear_Regression()
	trainingSet=ridge_regression.file2matrix("hw4_train.dat")
	testingSet=ridge_regression.file2matrix("hw4_test.dat")
	ridge_regression.iteration(trainingSet,10)
	Ein=ridge_regression.get0_1Error(trainingSet)
	Eout=ridge_regression.get0_1Error(testingSet)
	print("**************************************************************************")
	print("第13题答案如下：")
	print ('Ein:'+str(Ein))
	print ('Eout:'+str(Eout))
	print("**************************************************************************")
	print()

	Lamb=[i for i in range(-10,3)]
	min_Eout14=10
	min_Ein14=10
	min_Eout15=10
	min_Ein15=10
	lamb14=0
	lamb15=0
	for i in Lamb:
		new_i=10**i
		ridge_regression.w=np.zeros(3)
		ridge_regression.iteration(trainingSet,new_i)
		Ein=ridge_regression.get0_1Error(trainingSet)
		Eout=ridge_regression.get0_1Error(testingSet)
		if(Ein<=min_Ein14):
			min_Ein14=Ein
			min_Eout14=Eout
			lamb14=i
		if(Eout<=min_Eout15):
			min_Eout15=Eout
			min_Ein15=Ein
			lamb15=i

	print("**************************************************************************")
	print("第14题答案如下：")
	print("log10(lambda):"+str(lamb14))
	print('Ein:'+str(min_Ein14))
	print('Eout:'+str(min_Eout14))
	print("**************************************************************************")
	print()
	print("**************************************************************************")
	print("第15题答案如下：")
	print("log10(lambda):"+str(lamb15))
	print('Ein:'+str(min_Ein15))
	print('Eout:'+str(min_Eout15))
	print("**************************************************************************")
	print()


if __name__=="__main__":
	main()
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

	def slipDateSet(self,dataSet):
		m=len(dataSet)
		sp=int(m*0.6)
		trainingSet=dataSet[:sp,:]
		validationSet=dataSet[sp:,:]
		return trainingSet,validationSet

	def V_fold_cross_validation(self,dataSet,v,lamb):
		m=len(dataSet)
		Ecv=0
		W=np.zeros(3)
		for i in range(v):
			self.w=np.zeros(3)
			start=int(m*i/v)
			end=int(m*(i+1)/v)
			validationSet=dataSet[start:end,:]
			trainingSet=np.concatenate((dataSet[:start,:],dataSet[end:,:]),axis=0)
			self.iteration(trainingSet,lamb)
			Ecv+=self.get0_1Error(validationSet)
		return Ecv/5

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

	trainingSet,validationSet=ridge_regression.slipDateSet(trainingSet)
	min_Etrain16=10
	Eval16=10
	Eout16=10
	Etrain17=10
	min_Eval17=10
	Eout17=10
	lamb16=0
	lamb17=0
	tmpW=np.zeros(3)
	for i in Lamb:
		new_i=10**i
		ridge_regression.w=np.zeros(3)
		ridge_regression.iteration(trainingSet,new_i)
		Etrain=ridge_regression.get0_1Error(trainingSet)
		Eval=ridge_regression.get0_1Error(validationSet)
		Eout=ridge_regression.get0_1Error(testingSet)
		if(Etrain<=min_Etrain16):
			min_Etrain16=Etrain
			Eval16=Eval
			Eout16=Eout
			lamb16=i
		if(Eval<=min_Eval17):
			min_Eval17=Eval
			Etrain17=Etrain
			Eout17=Eout
			lamb17=i


	print("**************************************************************************")
	print("第16题答案如下：")
	print("log10(lambda):"+str(lamb16))
	print('Etrain:'+str(min_Etrain16))
	print('Eval:'+str(Eval16))
	print('Eout:'+str(Eout16))
	print("**************************************************************************")
	print()
	print("**************************************************************************")
	print("第17题答案如下：")
	print("log10(lambda):"+str(lamb17))
	print('Etrain:'+str(Etrain17))
	print('Eval:'+str(min_Eval17))
	print('Eout:'+str(Eout17))
	print("**************************************************************************")
	print()

	trainingSet=ridge_regression.file2matrix("hw4_train.dat")
	ridge_regression.iteration(trainingSet,10**lamb17)
	Ein=ridge_regression.get0_1Error(trainingSet)
	Eout=ridge_regression.get0_1Error(testingSet)
	print("**************************************************************************")
	print("第18题答案如下：")
	print ('Ein:'+str(Ein))
	print ('Eout:'+str(Eout))
	print("**************************************************************************")
	print()	

	min_Ecv=10
	lamb19=0
	for i in Lamb:
		lamb=10**i
		ave_Ecv=ridge_regression.V_fold_cross_validation(trainingSet,5,lamb)
		if(ave_Ecv<=min_Ecv):
			min_Ecv=ave_Ecv
			lamb19=i
	print("**************************************************************************")
	print("第19题答案如下：")
	print ("log10(lambda):"+str(lamb19))
	print ('Ecv:'+str(min_Ecv))
	print("**************************************************************************")
	print()		

	ridge_regression.iteration(trainingSet,10**lamb19)
	Ein=ridge_regression.get0_1Error(trainingSet)
	Eout=ridge_regression.get0_1Error(testingSet)
	print("**************************************************************************")
	print("第20题答案如下：")
	print ('Ein:'+str(Ein))
	print ('Eout:'+str(Eout))
	print("**************************************************************************")
	print()

if __name__=="__main__":
	main()
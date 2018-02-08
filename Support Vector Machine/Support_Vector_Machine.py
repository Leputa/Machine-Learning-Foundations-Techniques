import numpy as np
from sklearn import svm

class Support_Vector_Machine():
	def file2matrix(self,filename,target):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,3))
		index=0
		for line in lines:
			listFromLine=line.strip().split()
			for i in range(len(listFromLine)):
				listFromLine[i]=float(listFromLine[i])
			if listFromLine[0]==target:            #第一列是Y,后两列是X
				listFromLine[0]=1
			else:
				listFromLine[0]=-1
			dataSet[index]=listFromLine
			index+=1
		return dataSet	

	def linearKernel(self,trainingSet,C):
		#model=svm.libsvm.fit(X=trainingSet[:,1:].astype(np.float64),Y=np.transpose(trainingSet[:,:1]).astype(np.float64),kernel='linear',C=C)
		model=svm.SVC(C=C,kernel='linear')
		X=trainingSet[:,1:]
		y=trainingSet[:,:1].flatten()
		print(y)
		model.fit(X,y)
		w=model.coef_[0]
		return np.sqrt(np.sum(np.square(w)))

	def polynomialKernel(self,filename,C,Q):
		min_Ein=1
		tagMin=-1
		maxsumOfAn=-1
		for target in range(0,10,2):
			trainingSet=self.file2matrix(filename,target)
			model=svm.SVC(C=C,kernel='poly',degree=Q)
			X=trainingSet[:,1:]
			y=trainingSet[:,:1].flatten()
			model.fit(X,y)
			y_=model.predict(X)
			Ein=0
			for i in range(len(y)):
				if y[i]!=y_[i]:
					Ein+=1
			Ein=Ein/len(y)
			if Ein<min_Ein:
				min_Ein=Ein
				tagMin=target
			sumOfAn=np.sum(np.fabs(model.dual_coef_[0]))
			if(sumOfAn>maxsumOfAn):
				maxsumOfAn=sumOfAn
		return tagMin,Ein,maxsumOfAn

	def GaussianKernel(self,trainingSet,testingSet,C):
		Gamma=1
		minGamma=0
		minEout=8000
		while(Gamma<=10000):
			model=svm.SVC(C=C,kernel='rbf',gamma=Gamma)
			model.fit(trainingSet[:,1:],trainingSet[:,:1].flatten())
			X_=testingSet[:,1:]
			y_=testingSet[:,:1].flatten()
			y=model.predict(X_)
			Eout=0
			for i in range(len(y)):
				if y[i]!=y_[i]:
					Eout+=1
			if Eout<minEout:
				minEout=Eout
				minGamma=Gamma
			Gamma*=10
		return minGamma,minEout/len(testingSet)

	def crossValidation(self,trainingSet,C):
		Gamma=1
		minGamma=0
		minEval=8000
		while(Gamma<=1000):
			Eval=0
			model=svm.SVC(C=C,kernel='rbf',gamma=Gamma)
			for i in range(100):
				np.random.shuffle(trainingSet)
				model.fit(trainingSet[1000:,1:],trainingSet[1000:,:1].flatten())
				Xval=trainingSet[:1000,1:]
				Yval=trainingSet[:1000,:1].flatten()
				y_=model.predict(Xval)
				for i in range(len(y_)):
					if Yval[i]!=y_[i]:
						Eval+=1
			Eval=Eval/100
			if Eval<minEval:
				minEval=Eval
				minGamma=Gamma
			Gamma*=10
		return minGamma,Eval/1000

def main():
	SVM=Support_Vector_Machine()
	trainingSet=SVM.file2matrix("features.train.dat",0)
	lengthOfW=SVM.linearKernel(trainingSet,0.01)
	print("**************************************************************************")
	print("第15题答案如下：")
	print ('||W||:'+str(lengthOfW))
	print("**************************************************************************")
	print()
	target,Ein,numofSupportVector=SVM.polynomialKernel("features.train.dat",0.01,2)
	print("**************************************************************************")
	print("第16题答案如下：")
	print ('the SVM classifiers reaches the lowest Ein:'+str(target))
	print ('the lowest Ein:'+str(Ein))
	print("**************************************************************************")
	print()
	print("**************************************************************************")
	print("第17题答案如下：")
	print ('Sum of An:'+str(numofSupportVector))
	print("**************************************************************************")
	print()
	testingSet=SVM.file2matrix("features.test.dat",0)
	Gamma,Eout=SVM.GaussianKernel(trainingSet,testingSet,0.1)
	print("**************************************************************************")
	print("第19题答案如下：")
	print ('the Gamma reaches the lowest Eout:'+str(Gamma))
	print ('the lowest Eout:'+str(Eout))
	print("**************************************************************************")
	print()
	Gamma,Eval=SVM.crossValidation(trainingSet,0.1)
	print("**************************************************************************")
	print("第20题答案如下：")
	print ('the Gamma reaches the lowest Eval:'+str(Gamma))
	print ('the lowest Eval:'+str(Eval))
	print("**************************************************************************")
	print()

if __name__=="__main__":
	main()
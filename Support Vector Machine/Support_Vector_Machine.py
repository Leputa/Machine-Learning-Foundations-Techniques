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
		model.fit(X,y)
		w=model.coef_[0]
		return np.sqrt(np.sum(np.square(w)))

	def polynomialKernel(self,filename,C,Q):
		min_Ein=1
		tagMin=-1
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
		return tagMin,Ein




def main():
	SVM=Support_Vector_Machine()
	trainingSet=SVM.file2matrix("features.train.dat",0)
	lengthOfW=SVM.linearKernel(trainingSet,0.01)
	print("**************************************************************************")
	print("第15题答案如下：")
	print ('||W||:'+str(lengthOfW))
	print("**************************************************************************")
	print()
	target,Ein=SVM.polynomialKernel("features.train.dat",0.01,2)
	print("**************************************************************************")
	print("第16题答案如下：")
	print ('the SVM classifiers reaches the lowest Ein:'+str(target))
	print ('the lowest Ein:'+str(Ein))
	print("**************************************************************************")
	print()


if __name__=="__main__":
	main()
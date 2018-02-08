import numpy as np 

class K_Nearest_Neighbor():
	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,10))
		index=0
		for line in lines:                                                                                                       
			dataSet[index]=line.strip().split(' ')
			index+=1
		return dataSet

	def KNNError(self,trainingSet,testingSet,k=5):
		Error=0
		m=np.shape(testingSet)[0]
		for i in range(len(testingSet)):
			dist=np.sum((trainingSet[:,:-1]-testingSet[i,:-1])**2,axis=1)
			kIndex=np.argsort(dist)[:k]
			y_=0
			for index in kIndex:
				y_+=trainingSet[index,-1]
			y_/=k
			if np.sign(y_)!=testingSet[i,-1]:
				Error+=1
		return Error/m

def main():
	knn=K_Nearest_Neighbor()
	trainingSet=knn.file2matrix('hw4_knn_train.dat')
	testingSet=knn.file2matrix('hw4_knn_test.dat')
	Ein=knn.KNNError(trainingSet,trainingSet,1)
	print("**************************************************************************")
	print("第15题答案如下：")
	print ("Ein(gnbor):"+str(Ein))
	print("**************************************************************************")
	print()
	Eout=knn.KNNError(trainingSet,testingSet,1)
	print("**************************************************************************")
	print("第16题答案如下：")
	print ("Eout(gnbor):"+str(Eout))
	print("**************************************************************************")
	print()
	Ein=knn.KNNError(trainingSet,trainingSet,5)
	print("**************************************************************************")
	print("第17题答案如下：")
	print ("Ein(g5nbor):"+str(Ein))
	print("**************************************************************************")
	print()
	Eout=knn.KNNError(trainingSet,testingSet,5)
	print("**************************************************************************")
	print("第18题答案如下：")
	print ("Eout(g5nbor):"+str(Eout))
	print("**************************************************************************")
	print()

if __name__=="__main__":
	main()
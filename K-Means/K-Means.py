#K=10时有可能遇到1个类中一个点都没有

import numpy as np

class K_Means():
	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,9))
		index=0
		for line in lines:                                                                                                       
			dataSet[index]=line.strip().split(' ')
			index+=1
		return dataSet

	def KMeans(self,dataSet,k,T=5000):
		m,n=np.shape(dataSet)
		K=self.randomPickK(dataSet,k)
		for i in range(T):
			S=[[] for i in range(k)]
			for i in range(m):
				dist=np.sum((K-dataSet[i])**2,axis=1)
				index=np.argmin(dist)
				S[index].append(dataSet[i])
			K=np.zeros((k,n))
			for i in range(k):
				for s in S[i]:
					K[i]+=s
				if (len(S[i])==0):
					K[i]=np.random.randint(m)
				else:
					K[i]/=len(S[i])
		return K,S

	def calError(self,dataSet,K,S,k):
		m=np.shape(dataSet)[0]
		Error=0
		length=0
		for i in range(k):
			length+=len(S[i])
			Error+=np.sum((S[i]-K[i])**2)
		return Error/m

	def randomPickK(self,dataSet,k):
		m,n=np.shape(dataSet)
		K=np.zeros((k,n))
		Index=[]
		for i in range(k):
			index=np.random.randint(0,m)
			while(index in Index):
				index=np.random.randint(0,m)
			K[i]=dataSet[index]	
		return K	


def main():
	k_means=K_Means()
	dataSet=k_means.file2matrix('hw4_kmeans_train.dat')
	K,S=k_means.KMeans(dataSet,2)
	Ein=k_means.calError(dataSet,K,S,2)
	print("**************************************************************************")
	print("第19题答案如下：")
	print ("Ein of 2-Means:"+str(Ein))
	print("**************************************************************************")
	print()
	K,S=k_means.KMeans(dataSet,10)
	Ein=k_means.calError(dataSet,K,S,10)
	print("**************************************************************************")
	print("第20题答案如下：")
	print ("Ein of 10-Means:"+str(Ein))
	print("**************************************************************************")
	print()

if __name__=='__main__':
	main()

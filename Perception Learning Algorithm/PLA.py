import numpy as np
import operator
import random

class PLA():

	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,5))
		index=0
		for line in lines:
			line=line.strip()
			listFromLine=line.split('\t')
			numList=listFromLine[0].split(' ')
			numList.append(' '+listFromLine[-1])
			dataSet[index]=numList
			index+=1
		one=np.ones(m)
		dataSet=np.insert(dataSet, 0, values=one, axis=1)
		return dataSet

	def classfy(self,dataSet,isFifteen):
		w=np.random.randn(5)
		cnt=0
		while (True):
			flag=True
			for data in dataSet:
				if np.dot(w,data[:5])*data[-1]>=0:
					continue
				else:
					cnt+=1
					flag=False
					w=w+data[:5]*data[-1]
			if(flag):
				break
		if(isFifteen):
			print("**************************************************************************")
			print("第15题答案如下：")
			print ('迭代次数:'+str(cnt))
			print ('更新后权重:'+str(w))
			print("**************************************************************************")
			print()
		return cnt

	def classfy_a(self,dataSet,a):
		w=np.random.randn(5)
		cnt=0
		while (True):
			flag=True
			for data in dataSet:
				if np.dot(w,data[:5])*data[-1]>=0:
					continue
				else:
					cnt+=1
					flag=False
					w=w+a*data[:5]*data[-1]
			if(flag):
				break
		return cnt

	def AverageIteration(self,dataSet,number):
		sumCount=0
		sumCount_a=0
		for i in range(number):
			np.random.shuffle(dataSet)
			sumCount+=self.classfy(dataSet,False)
			sumCount_a+=self.classfy_a(dataSet,0.5)
		print("**************************************************************************")
		print("第16题答案如下：")
		print("平均迭代次数："+str(sumCount//number))
		print("**************************************************************************")
		print()

		print("**************************************************************************")
		print("第17题答案如下：")
		print("平均迭代次数："+str(sumCount_a//number))
		print("**************************************************************************")
		print()



def main():
	pla=PLA()
	dataSet=pla.file2matrix('hw1_15_train.dat')
	pla.classfy(dataSet,True)
	pla.AverageIteration(dataSet,2000)
	 
if __name__=='__main__':
	main()

import numpy as np
import copy

class PA():
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

	def classfy(self,dataSet,eta,number):
		w=np.zeros(5)
		paW=np.random.randn(5)
		iter=0
		errorRate=1
		m=len(dataSet)
		if(number==18 or number==19):
			iteration=50
		elif(number==20):
			iteration=100
		while (iter<iteration):
			for data in dataSet:
				if np.dot(paW,data[:5])*data[-1]<=0:
					paW=paW+data[:5]*data[-1]*eta
					if(number==18 or number==20):
						paErrorRate=self.getErrorRate(dataSet,paW,m)
						if(paErrorRate<errorRate):
							w=paW
							errorRate=paErrorRate
					elif(number==19):
						w=paW
					iter+=1				
		return w

	def getErrorRate(self,dataSet,w,m):
		error=0
		for data in dataSet:
			if np.dot(w,data[:5])*data[-1]<=0:
				error+=1
		return error/m

	def validation(self,TrainingSet,TestingSet,number):
		m=len(TestingSet)
		num=0
		error_sum=0.0
		while(num<2000):
			if(num%100)==0:
				print('当前实验次数：'+str(num))
			np.random.shuffle(TrainingSet)
			w=self.classfy(TrainingSet,0.5,number)
			errorRate=self.getErrorRate(TestingSet,w,m)
			num+=1
			error_sum+=errorRate
		if(number==18):
			print("**************************************************************************")
			print("第18题答案如下：")
			print ('平均错误率:'+str(error_sum/num))
			print("**************************************************************************")
			print()
		elif(number==19):
			print("**************************************************************************")
			print("第18题答案如下：")
			print ('平均错误率:'+str(error_sum/num))
			print("**************************************************************************")
			print()
		elif(number==20):
			print("**************************************************************************")
			print("第20题答案如下：")
			print ('平均错误率:'+str(error_sum/num))
			print("**************************************************************************")
			print()
			


def main():
	pa=PA()
	trainingSet=pa.file2matrix('hw1_18_train.dat')
	testingSet=pa.file2matrix('hw1_18_test.dat')
	pa.validation(trainingSet,testingSet,18)
	pa.validation(trainingSet,testingSet,19)
	pa.validation(trainingSet,testingSet,20)
	 
if __name__=='__main__':
	main()



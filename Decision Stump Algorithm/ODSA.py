import numpy as np
import random

class One_Dimensional_Decision_Stump_Algorithm():
	def __init__(self):
		self.dataSize=20
		self.testNum=5000
		self.s=0
		self.theta=0

	def file2matrix(self):
		dataSet=np.zeros((self.dataSize,2))
		for data in dataSet:
			data[0]=random.uniform(-1,1)
			data[1]=np.sign(data[0])  #默认大于0是圈圈，小于0是叉叉
		dataSet=dataSet[dataSet[:,0].argsort()]
		return self.addNoise(dataSet)

	def addNoise(self,dataSet):
		for data in dataSet:
			rand=random.random()
			if rand<=0.2:
				data[1]=-data[1]
		return dataSet

	def E_in(self):
		trainingSet=self.file2matrix()
		theta=0
		min_errorRate=1.0
		for i in range(self.dataSize+1):
			s=1
			if (i==0):
				theta=trainingSet[0][0]-1.0
			elif (i==self.dataSize):
				theta=trainingSet[self.dataSize-1][0]+1.0
			else:
				theta=(trainingSet[i-1][0]+trainingSet[i][0])/2
			errorRate=self.calculateError(trainingSet,s,theta)
			if(errorRate<min_errorRate):
				self.s=s
				self.theta=theta
				min_errorRate=errorRate
			s=-1
			errorRate=self.calculateError(trainingSet,s,theta)
			if(errorRate<min_errorRate):
				self.s=s
				self.theta=theta
				min_errorRate=errorRate
		return min_errorRate

	def E_out(self):
		testingSet=self.file2matrix()
		return self.calculateError(testingSet,self.s,self.theta)

	def calculateError(self,dataSet,s,theta):
		error=0
		for data in dataSet:
			if(s*np.sign(data[0]-theta)!=data[1]):
				error+=1
		return error/self.dataSize

	def calculateAverage(self):
		Ein=0
		Eout=0
		for i in range(self.testNum):
			Ein+=self.E_in()
			Eout+=self.E_out()
		return Ein/self.testNum,Eout/self.testNum


def main():
	oddsa=One_Dimensional_Decision_Stump_Algorithm()
	Ein,Eout=oddsa.calculateAverage()
	print("Ein:"+str(Ein))
	print("Eout:"+str(Eout))
	
if __name__=='__main__':
	main()

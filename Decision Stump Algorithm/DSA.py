import numpy as np

class Decision_Stump_Algorithm():
	def __init__(self):
		self.s=0
		self.theta=0

	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,10))
		index=0
		for line in lines:
			line=line.strip()
			listFromLine=line.split('\t')
			numList=listFromLine[0].split(' ')
			dataSet[index]=numList
			index+=1
		return dataSet



	def E_in(self,trainingSet):
		theta=[0]*9
		min_errorRate=1.0
		m=len(trainingSet)
		for k in range(9):
			trainingSet=trainingSet[trainingSet[:,k].argsort()]
			for i in range(m+1):
				if i==0:
					theta=trainingSet[0][k]-1.0
				elif i==m:
					theta=trainingSet[m-1][k]+1.0
				else:
					theta=(trainingSet[i-1][k]+trainingSet[i][k])/2
				s=1
				errorRate=self.calculateError(trainingSet,s,theta,k,m)
				if(errorRate<min_errorRate):
					self.theta=theta
					self.s=s
					min_errorRate=errorRate
				s=-1
				errorRate=self.calculateError(trainingSet,s,theta,k,m)
				if(errorRate<min_errorRate):
					self.theta=theta
					self.s=s
					min_errorRate=errorRate
		return min_errorRate

	def E_out(self,Testingfile):
		min_errorRate=1.0
		m=len(Testingfile)
		for k in range(9):
			errorRate=self.calculateError(Testingfile,self.s,self.theta,k,m)
			if(errorRate<min_errorRate):
				min_errorRate=errorRate
		return min_errorRate

	def calculateError(self,dataSet,s,theta,k,m):
		error=0
		for data in dataSet:
			if(s*np.sign(data[k]-theta)!=data[-1]):
				error+=1
		return error/m



def main():
	dsa=Decision_Stump_Algorithm()
	trainingSet=dsa.file2matrix('hw2_train.dat')
	TestingSet=dsa.file2matrix('hw2_test.dat')
	Ein=dsa.E_in(trainingSet)
	Eout=dsa.E_out(TestingSet)
	print("**************************************************************************")
	print("第19题答案如下：")
	print("Ein:"+str(Ein))
	print("**************************************************************************")
	print()
	print("**************************************************************************")
	print("第20题答案如下：")
	print("Eout:"+str(Eout))
	print("**************************************************************************")	
if __name__=="__main__":
	main()

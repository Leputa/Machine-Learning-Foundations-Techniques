import numpy as np
import math

class AdaBoost_Stump_Algorithm():
	def file2matrix(self,filename,iterations=300):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,3))
		index=0
		for line in lines:
			line=line.strip()
			listFromLine=line.split(' ')
			dataSet[index]=listFromLine
			index+=1
		u=np.zeros((iterations,m))   
		u+=1/m
		return dataSet,u

	def trainAlphaOnTheFly(self,trainingSet,u,iterations=300):
		alpha=np.zeros(iterations)
		theta=np.zeros(iterations)
		s=np.zeros(iterations)
		dimension=np.zeros(iterations)
		m=len(trainingSet)
		minE=1.0
		for i in range(iterations):
			minErrorRate=1.0
			for k in range(2):
				trainingSet=trainingSet[trainingSet[:,k].argsort()]
				for j in range(m+1):
					if j==0:
						tmpTheta=trainingSet[0][k]-1.0
					elif j==m:
						tmpTheta=trainingSet[m-1][k]+1.0
					else:
						tmpTheta=(trainingSet[j-1][k]+trainingSet[j][k])/2
					tmpS=1
					errorRate=self.calculateError(trainingSet,tmpS,tmpTheta,k,u[i])
					if errorRate<minErrorRate:
						minErrorRate=errorRate
						s[i]=tmpS
						theta[i]=tmpTheta
						dimension[i]=k
					tmpS=-1
					errorRate=self.calculateError(trainingSet,tmpS,tmpTheta,k,u[i])
					if errorRate<minErrorRate:
						minErrorRate=errorRate
						s[i]=tmpS
						theta[i]=tmpTheta
						dimension[i]=k
			if(minErrorRate<minE):
				minE=minErrorRate
			diamond=math.sqrt((1 - minErrorRate)/minErrorRate)
			alpha[i]=math.log(diamond)
			if i!=iterations-1:
				self.updataU(trainingSet,theta,s,u,i,dimension[i],diamond)
			if (i==0):
				self.print12(minErrorRate)
				self.print13(u[i+1])
			elif(i==iterations-2):
				self.print15(u[i+1])
		self.print16(minE)
		return alpha,theta,s,dimension

	def updataU(self,trainingSet,theta,s,u,i,dimension,diamond):
		for j in range(len(trainingSet)):
			if s[i]*np.sign(trainingSet[j][int(dimension)]-theta[i])==trainingSet[j][-1]:
				u[i+1][j]=u[i][j]/diamond
			else:
				u[i+1][j]=u[i][j]*diamond

	def calculateError(self,dataSet,s,theta,k,u):
		error=0
		for i in range(len(dataSet)):
			if(s*np.sign(dataSet[i][k]-theta)!=dataSet[i][-1]):
				error+=1*u[i]
		return error	

	def calculateGError(self,dataSet,alpha,theta,s,dimension,u,iterations=300):
		GError=0
		m=len(dataSet)
		for i in range(m):
			G=0
			for t in range(iterations):
				G+=np.sign(alpha[t]*(s[t]*np.sign(dataSet[i][int(dimension[t])]-theta[t])))
			if (np.sign(G)!=dataSet[i][-1]):
				GError+=1
		return GError/len(dataSet)

	def print12(self,Ein):
		print("**************************************************************************")
		print("第12题答案如下：")
		print("Ein(g1):"+str(Ein))
		print("**************************************************************************")
		print()	

	def print13(self,u):
		print("**************************************************************************")
		print("第14题答案如下：")
		print("U2:"+str(np.sum(u)))
		print("**************************************************************************")
		print()		

	def print15(self,u):
		print("**************************************************************************")
		print("第15题答案如下：")
		print("UT:"+str(np.sum(u)))
		print("**************************************************************************")
		print()		

	def print16(self,minE):
		print("**************************************************************************")
		print("第16题答案如下：")
		print("minimum value of eta:"+str(minE))
		print("**************************************************************************")
		print()			

	def print18(self,Eout):
		print("**************************************************************************")
		print("第18题答案如下：")
		print("Eout:"+str(minE))
		print("**************************************************************************")
		print()

def main():
	adaBoost=AdaBoost_Stump_Algorithm()
	trainingSet,u=adaBoost.file2matrix('hw2_adaboost_train.dat')
	testingSet,utest=adaBoost.file2matrix('hw2_adaboost_test.dat')
	alpha,theta,s,dimension=adaBoost.trainAlphaOnTheFly(trainingSet,u)
	Eout1=adaBoost.calculateError(testingSet,s[0],theta[0],int(dimension[0]),utest[0])
	print("**************************************************************************")
	print("第17题答案如下：")
	print("Eout(g1):"+str(Eout1))
	print("**************************************************************************")
	print()
	EinG=adaBoost.calculateGError(trainingSet,alpha,theta,s,dimension,u)
	print("**************************************************************************")
	print("第13题答案如下：")
	print("Ein(G):"+str(EinG))
	print("**************************************************************************")
	print()	
	EoutG=adaBoost.calculateGError(testingSet,alpha,theta,s,dimension,utest)
	print("**************************************************************************")
	print("第18题答案如下：")
	print("Eout(G):"+str(EoutG))
	print("**************************************************************************")
	print()	

if __name__=="__main__":
	main()
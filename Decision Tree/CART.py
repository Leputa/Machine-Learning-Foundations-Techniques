import numpy as np

class Decision_Tree():
	def __init__(self):
		self.NumOfInternalNode=0

	def file2matrix(self,filename):
		fr=open(filename)
		lines=fr.readlines()
		m=len(lines)
		dataSet=np.zeros((m,3))
		index=0
		for line in lines:
			lineList=line.strip().split(' ')
			dataSet[index]=lineList
			index+=1
		return dataSet

	def binSplitDataSet(self,dataSet,feature,value):   #左1右-1
		subDataSet0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
		subDataSet1=dataSet[np.nonzero(dataSet[:,feature]<=value)[0],:]
		return subDataSet0,subDataSet1

	def chooseBestToSplit(self,trainingSet):
		if(len(set(trainingSet[:,-1].tolist()))==1):
			return None,trainingSet[0,-1]
		self.NumOfInternalNode+=1
		m,n=np.shape(trainingSet)
		minGini=np.inf
		minFeat=0
		minVal=0
		for feature in range(n-1):
			for splitVal in set(trainingSet[:,feature].tolist()):
				subDataSet0,subDataSet1=self.binSplitDataSet(trainingSet,feature,splitVal)
				Gini=self.computeGini(trainingSet,subDataSet0,subDataSet1)
				if Gini<minGini:
					minGini=Gini
					minFeat=feature
					minVal=splitVal
		return minFeat,minVal

	def computeGini(self,trainingSet,subDataSet0,subDataSet1):
		Gini0=self.Gini(subDataSet0)
		Gini1=self.Gini(subDataSet1)
		m=len(trainingSet)
		m0=len(subDataSet0)
		m1=len(subDataSet1)
		return Gini0*m0/m+Gini1*m1/m

	def Gini(self,dataSet):
		m=len(dataSet)
		if m==0:
			return 0
		y=0
		for label in dataSet[:,-1]:
			if label==1.0:
				y+=1
		return 2*y/m*(1-y/m)

	def createTree(self,trainingSet):
		feature,s=self.chooseBestToSplit(trainingSet)
		if feature==None:
			return s
		binTree={}
		binTree['feature']=feature
		binTree['value']=s
		leftSet,rightSet=self.binSplitDataSet(trainingSet,feature,s)
		binTree['left']=self.createTree(leftSet)
		binTree['right']=self.createTree(rightSet)
		return binTree

	def createPruneTree(self,trainingSet):
		feature,s=self.chooseBestToSplit(trainingSet)
		binTree={}
		binTree['feature']=feature
		binTree['value']=s
		leftSet,rightSet=self.binSplitDataSet(trainingSet,feature,s)
		left=np.sign(np.sum(leftSet[:,-1]))
		right=np.sign(np.sum(rightSet[:,-1]))
		binTree['left']=left
		binTree['right']=right
		return binTree

	def calError(self,dataSet,dtree):
		m=len(dataSet)
		yHat=np.zeros((m,1))
		error=0
		for i in range(m):
			yHat[i,0]=self.calErrorDFS(dataSet[i],dtree)
			if (yHat[i,0]!=dataSet[i,-1]):
				error+=1
		return error/m,yHat

	def calErrorDFS(self,data,dtree):
		if (type(dtree).__name__!='dict'):
			return dtree
		if data[dtree['feature']]>dtree['value']:
			if (type(dtree['left']).__name__!='dict'):
				return dtree['left']
			else:
				return self.calErrorDFS(data,dtree['left'])
		else:
			if (type(dtree['right']).__name__!='dict'):
				return dtree['right']	
			else:
				return self.calErrorDFS(data,dtree['right'])

class RandomForest():

	def createForest(self,trainingSet,T):
		m=len(trainingSet)
		RF=[]
		RFDS=[]
		for i in range(T):
			trainingSet_=self.bootStrapping(trainingSet,m)
			dt=Decision_Tree()
			tree=dt.createTree(trainingSet_)
			RF.append(tree)
			RFDS.append(dt)
		return RF,RFDS

	def createPruneForest(self,trainingSet,T):
		m=len(trainingSet)
		RF=[]
		RFDS=[]
		for i in range(T):
			trainingSet_=self.bootStrapping(trainingSet,m)
			dt=Decision_Tree()
			tree=dt.createPruneTree(trainingSet_)
			RF.append(tree)
			RFDS.append(dt)
		return RF,RFDS

	def bootStrapping(self,trainingSet,m):
		trainingSet_=np.zeros((m,3))
		for i in range(m):
			i_=np.random.randint(m)
			trainingSet_[i]=trainingSet[i_]
		return trainingSet_

	def calError(self,dataSet,RF,RFDS):
		errorg=0
		errorG=0
		m=len(dataSet)
		Y=np.zeros((len(RF),m))
		y=dataSet[:,-1]
		y_=np.zeros(m)
		for i in range(len(RF)):
			dt=RFDS[i]
			tree=RF[i]
			tmpErrorg,yHat=dt.calError(dataSet,tree)
			errorg+=tmpErrorg
			Y[i,:]=yHat.transpose()[0]
		for i in range(m):
			y_[i]=np.sign(np.sum(Y[:,i]))
			if y_[i]!=y[i]:
				errorG+=1
		return errorg/len(RF),errorG/m

def RFevarageError(trainingSet,testingSet,iteration):
	errorg=0
	errorG=0
	errorOutG=0
	for i in range(iteration):
		RF=RandomForest()
		forest,forestDS=RF.createForest(trainingSet,300)
		errorIng,errorInG=RF.calError(trainingSet,forest,forestDS)
		errorg+=errorIng
		errorG+=errorInG
		errorOutg,tmpErrorOutG=RF.calError(testingSet,forest,forestDS)
		errorOutG+=tmpErrorOutG
	return errorg/iteration,errorG/iteration,errorOutG/iteration

def RFevarageErrorByPrune(trainingSet,testingSet,iteration):
	errorG=0
	errorOutG=0
	for i in range(iteration):
		RF=RandomForest()
		forest,forestDS=RF.createPruneForest(trainingSet,300)
		errorIng,errorInG=RF.calError(trainingSet,forest,forestDS)
		errorG+=errorInG
		errorOutg,tmpErrorOutG=RF.calError(testingSet,forest,forestDS)
		errorOutG+=tmpErrorOutG
	return errorG/iteration,errorOutG/iteration

def main():
	dt=Decision_Tree()
	trainingSet=dt.file2matrix('hw3_train.dat')
	binTree=dt.createTree(trainingSet)

	print("**************************************************************************")
	print("第13题答案如下：")
	print ('Number of Internal Nodes:'+str(dt.NumOfInternalNode))
	print("**************************************************************************")
	print()
	Ein,yHat=dt.calError(trainingSet,binTree)
	print("**************************************************************************")
	print("第14题答案如下：")
	print ('Ein:'+str(Ein))
	print("**************************************************************************")
	print()
	testingSet=dt.file2matrix('hw3_test.dat')
	Eout,yHat=dt.calError(testingSet,binTree)
	print("**************************************************************************")
	print("第15题答案如下：")
	print ('Eout:'+str(Eout))
	print("**************************************************************************")
	print()

	errorIng,errorInG,errorOutG=RFevarageError(trainingSet,testingSet,100)
	# errorIng,errorInG,errorOutG=RFevarageError(trainingSet,testingSet,1)
	print("**************************************************************************")
	print("第16题答案如下：")
	print ('average Ein(gt):'+str(errorIng))
	print("**************************************************************************")
	print()

	print("**************************************************************************")
	print("第17题答案如下：")
	print ('average Ein(Grf):'+str(errorInG))
	print("**************************************************************************")
	print()

	print("**************************************************************************")
	print("第18题答案如下：")
	print ('average Eout(Grf):'+str(errorOutG))
	print("**************************************************************************")
	print()

	# errorPruneInG,errorPruneOutG=RFevarageErrorByPrune(trainingSet,testingSet,1)
	errorPruneInG,errorPruneOutG=RFevarageErrorByPrune(trainingSet,testingSet,100)
	print("**************************************************************************")
	print("第19题答案如下：")
	print ('average Ein(Grf):'+str(errorPruneInG))
	print("**************************************************************************")
	print()

	print("**************************************************************************")
	print("第20题答案如下：")
	print ('average Eout(Grf):'+str(errorPruneOutG))
	print("**************************************************************************")
	print()
	
if __name__=="__main__":
	main()

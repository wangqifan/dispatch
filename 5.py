import numpy as np
import random
import matplotlib.pyplot as plt
import math


class  PSO():
    # ---------------pos-----------参数设置
    def __init__(self,PN,dim,max_iter):
        self.w=0.8
        self.c1=2
        self.c2=2
        self.r1=0.6
        self.r2=0.3
        self.pN=PN                            #粒子数量
        self.dim=dim                          #搜索维度
        self.max_iter=max_iter                #迭代次数
        self.X=np.zeros((self.pN,self.dim))   #所有粒子的位置
        self.V=np.zeros((self.pN,self.dim))   #所有粒子的速度
        self.pbest=np.zeros((self.pN,self.dim))#当前最佳位置
        self.gbest=np.zeros((1,self.dim))      #全局最佳位置
        self.p_fit=np.zeros(self.pN)           #每个个体历史最佳适应值
        self.fit=10e10;
        self.costcalclatefuns=[]               #费用计算函数
        self.presure=np.zeros((self.dim,2))    #水压范围
        self.capity=800                        #供水水量
        self.capitytopressurefuns=[]           #水量转换到水压

    #----------目标函数------
    def fitnessFunc(self,x):
        length=len(x)
        sum=0;
        for i in range(length):
            sum+=self.costcalclatefuns[i](x[i])
        return sum

    #-------淘汰函数----------
    def isvaild(self,x):
        if not self.isenough(x):
            return False
        if not self.isPressureRight(x):
            return False
        return True

    #--------水压合适判定-------
    def isPressureRight(self,X):
        length=len(X)
        for i in range(length):
            val=self.changetopressure(X[i],i)
            if val<self.presure[i][0] or val > self.presure[i][1]:
                return False
        return True

    #--------根据水量获取水压----

    def changetopressure(self,c,index):
        return self.capitytopressurefuns[index](c)

    #--------水量合适判定------
    def isenough(self,x):
        length=len(x)
        sum=0
        for  i in range(length):
            sum+=x[i]
        return sum>=self.capity

    #-----------初始化种群------------
    #--------清洗数据------
    def cleardata(self):
        #self.p_fit = np.zeros(self.pN)  # 每个个体历史最佳适应值
        #self.V = np.zeros((self.pN, self.dim))  # 所有粒子的速度
        #self.X=np.zeros((self.pN,self.dim))   #所有粒子的位置
        #self.pbest = np.zeros((self.pN, self.dim))  # 当前最佳位置
        temp_X=[]
        temp_V=[]
        temp_pbest=[]
        temp_fit=[]
        for i in range(self.pN):
            if  self.isvaild(self.X[i]):
                temp_X.append(self.X[i])
                temp_V.append(self.V[i])
                temp_pbest.append(self.pbest[i])
                temp_fit.append(self.p_fit[i])
        self.pN=len(temp_X)
        self.X=temp_X
        self.V=temp_V
        self.pbest=temp_pbest
        self.p_fit=temp_fit
    #-----水费计算函数-----
    def creatercalculate(self):
        k=2+random.uniform(0,1);
        def calculater(x):
            return x*k;
        return calculater
    #------水量水压转换------
    def createrchange(self):
        k=random.uniform(0.1,0.15)
        def capitytopressur(c):
            return c * k
        return capitytopressur
    def init_population(self):
        for i in range(self.dim):
            self.costcalclatefuns.append(self.creatercalculate())
            self.capitytopressurefuns.append(self.createrchange())
            #设置水压上下限制
            self.presure[i][0]+=8+random.randint(-1,1)
            self.presure[i][1]+=16+random.randint(-1,1)
        for i  in range(self.pN):
            for j in range(self.dim):
                self.X[i][j]=100+random.randint(-10,10)
                self.V[i][j]=random.uniform(-0.1,0.1)
            self.pbest[i]=self.X[i]
            tmp=self.fitnessFunc(self.X[i])
            self.p_fit[i]=tmp
            if self.isvaild(self.X[i]) and tmp<self.fit:
                self.fit=tmp
                self.gbest=self.X[i]
        self.cleardata()

    #-----------PSO_NIW----------
    def getpsoniw(self,i):
        tmax=0.8
        tend=0.2
        k=3.0
        self.w=math.exp(-k*pow(i/self.max_iter,2))*(tmax-tend)+tend
    #-----------学习因子修改-----
    def getweightofstudy(self,i):
        cstart=2.5
        cend=0.5;
        self.c1=cstart+(cend-cstart)*i/self.max_iter
        self.c2=cstart-self.c1

    #------------修改参数-----
    def parameterchange(self,i):
        self.getpsoniw(i)
        self.getweightofstudy(i)

    #--------------更新粒子位置-------
    def iterator(self):
        fitness=[]
        for t in range(self.max_iter):    #更新gbest pbest
            self.cleardata()
            print(self.pN)#清洗数据
            print(self.gbest)
            print(self.isvaild(self.gbest))
            for i  in range(self.pN):
                #temp=self.function(self.X[i])
                temp=self.fitnessFunc(self.X[i])
                if temp<self.p_fit[i] and self.isvaild(self.X[i]):
                    self.p_fit[i]=temp
                    self.pbest[i]=self.X[i]
                    if self.p_fit[i]<self.fit:
                        self.gbest=self.X[i]
                        self.fit=self.p_fit[i]
            self.parameterchange(t)
            for i in range(self.pN):
                self.V[i]=self.w*self.V[i]+self.c1*self.r1*(self.pbest[i]-self.X[i])+self.c2*self.r2*(self.gbest-self.X[i])
                self.X[i]=self.X[i]+self.V[i]
            fitness.append(self.fit)
        return fitness
#---------------------程序执行----------
def run():
    my_pso=PSO(PN=10000,dim=8,max_iter=30)
    my_pso.init_population()
    fitness=my_pso.iterator()
    plt.figure(1)
    plt.title("1")
    plt.xlabel("iterators")
    plt.ylabel("fitness")
    t=np.array([t for t in  range(0,my_pso.max_iter)])
    fitness=np.array(fitness)
    plt.plot(t,fitness,color='b',linewidth=3)
    plt.show()



if __name__ =='__main__':
    print("start")
    run()

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from functions import sphere,rosenbrock,rastrigrin,griewank,schaffer,levy



def sigmoid(x):
    value=-0.5*x
    if abs(value)>100:
        value/=100
    val=1/(1+np.exp(value))
    val-=0.5
    if val>0:
        return val+0.2
    else:
        return val-0.2


class  PSO():
    # ---------------pos-----------参数设置
    def __init__(self,PN,dim,max_iter):
        self.c1=2
        self.c2=2
        self.pN=PN                            #粒子数量
        self.dim=dim                          #搜索维度
        self.max_iter=max_iter                #迭代次数
        self.X=np.zeros((self.pN,self.dim))   #所有粒子的位置
        self.V=np.zeros((self.pN,self.dim))   #所有粒子的速度
        self.pbest=np.zeros((self.pN,self.dim))#当前最佳位置
        self.gbest=np.zeros((1,self.dim))      #全局最佳位置
        self.w=np.ones(self.pN)                #权重系数
        self.p_fit=np.zeros(self.pN)           #每个个体历史最佳适应值
        self.fit=1e10

    #----------目标函数--griewank----
    def fitnessFunc(self, x):
         return rosenbrock(x)


    #----------
    def lengthofvector(self,x):
        lenth=len(x)
        sum=0
        x=x**2
        for i in range(lenth):
            sum+=x[i]
        return sum**0.5

    #-----------初始化种群------------

    def init_population(self):
        for i  in range(self.pN):
            for j in range(self.dim):
                self.X[i][j]=random.uniform(-10,10)
                self.V[i][j]=random.uniform(-1,1)
            self.pbest[i]=self.X[i]
            tmp=self.fitnessFunc(self.X[i])
            self.p_fit[i]=tmp
            if tmp<self.fit:
                self.fit=tmp
                self.gbest=self.X[i]
    #-----------学习因子修改-----
    def getweightofstudy(self,i):
        cstart=2.5
        cend=0.5
        self.c1=cstart+(cend-cstart)*i/self.max_iter
        self.c2=cstart-self.c1

    #------------修改参数-----
    def parameterchange(self,i):
        self.getweightofstudy(i)
    #------------
    def scatter(self,i):
        plt.figure(i)
        color = ['r', 'y', 'k', 'g', 'm']
        plt.xlim((-2,2))
        plt.ylim((-2,2))
        plt.xlabel("x", size=14)
        plt.ylabel("y", size=14)
        plt.scatter(self.X[:,0],self.X[:,1],c=color,linewidths=1)
        plt.savefig(str(i)+".png")
        plt.close('all')

    #--------------更新粒子位置-------
    def iterator(self):
        fitness=[]
        for t in range(self.max_iter):           #更新gbest pbest
            for i  in range(self.pN):
                temp=self.fitnessFunc(self.X[i])
                if(temp<self.p_fit[i]):
                    self.p_fit[i]=temp
                    self.pbest[i]=self.X[i]
                    if self.p_fit[i]<self.fit:
                        self.gbest=self.X[i]
                        self.fit=self.p_fit[i]
            self.parameterchange(t)
            for i in range(self.pN):
                self.V[i]=0.9*self.w[i]*self.V[i]+0.6*self.c1*(self.pbest[i]-self.X[i])+0.3*self.c2*(self.gbest-self.X[i])
                temppre=self.fitnessFunc(self.X[i])
                self.X[i]=self.X[i]+self.V[i]
                templast=self.fitnessFunc(self.X[i])
                self.w[i]=0.5*sigmoid((temppre-templast)/self.lengthofvector(self.V[i]))
            fitness.append(self.fit)
            print(t)
            print(self.fit)
        return fitness
#---------------------程序执行----------
def run1():
    my_pso=PSO(PN=100,dim=200,max_iter=100)
    my_pso.init_population()
    fitness=my_pso.iterator()
    plt.figure(1)
    plt.title("1")
    plt.xlabel("iterators",size=14)
    plt.ylabel("fitness",size=14)
    plt.xlim(-1, 100)
    plt.ylim(-1,100000)
    t=np.array([t for t in  range(0,my_pso.max_iter)])
    fitness=np.array(fitness)
    print(len(t),"   ",len(fitness))
    plt.plot(t,fitness,color='g',linewidth=3)


if __name__ =='__main__':
    print("start")
    run1()

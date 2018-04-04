import numpy as np
import random
import matplotlib.pyplot as plt


#---------------pos-----------参数设置

class  PSO():
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
        self.fit=1e10;

    #-----------目标函数----------
    def function(self,x):
        sum=0
        length=len(x)
        x=x**2
        for i in range(length):
            sum+=x[i]
        return sum

    #-----------初始化种群------------
    def init_population(self):
        for i  in range(self.pN):
            for j in range(self.dim):
                self.X[i][j]=random.uniform(0,1)
                self.V[i][j]=random.uniform(0,1)
            self.pbest[i]=self.X[i]
            tmp=self.function(self.X[i])
            self.p_fit[i]=tmp
            if tmp<self.fit:
                self.fit=tmp
                self.gbest=self.X[i]

    #-----------线性递减策略---------
    def getweightoflinear(self,i):
        self.w=self.w-i*(0.8-0.4)/self.max_iter
    #----------线性微分递减策略-------
    def getweightofder(self,i):
        wstart=0.8
        wend=0.4
        self.w=wstart-(wstart-wend)/(self.max_iter*self.max_iter)*i*i
    #--------------更新粒子位置-------
    def iterator(self):
        fitness=[]
        for t in range(self.max_iter):           #更新gbest pbest
            for i  in range(self.pN):
                temp=self.function(self.X[i])
                if(temp<self.p_fit[i]):
                    self.p_fit[i]=temp
                    self.pbest[i]=self.X[i]
                    if self.p_fit[i]<self.fit:
                        self.gbest=self.X[i]
                        self.fit=self.p_fit[i]
            #self.getweightoflinear(t)
            self.getweightofder(t)
            for i in range(self.pN):
                self.V[i]=self.w*self.V[i]+self.c1*self.r1*(self.pbest[i]-self.X[i])+self.c2*self.r2*(self.gbest-self.X[i])
                self.X[i]=self.X[i]+self.V[i]
            fitness.append(self.fit)
            print(self.fit)
        return fitness
#---------------------程序执行----------
def run():
    my_pso=PSO(PN=30,dim=5,max_iter=20)
    my_pso.init_population()
    fitness=my_pso.iterator()

    plt.figure(1)
    plt.title("图1")
    plt.xlabel("iterators",size=14)
    plt.ylabel("fitness",size=14)
    t=np.array([t for t in  range(0,my_pso.max_iter)])
    fitness=np.array(fitness)
    print(len(t),"   ",len(fitness))
    plt.plot(t,fitness,color='b',linewidth=3)
    plt.show()



if __name__ =='__main__':
    print("start")
    run()

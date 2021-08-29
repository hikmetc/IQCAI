#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:15:15 2020

@author: hikmetcancubukcu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 23:39:42 2020

@author: hikmetcancubukcu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# veri girişi kodu yazılacak!!!





""" sınırı geçerse hata verip listeye atma kodu """
# Westgard EWMA CUSUM değerlendirme

class IQC_models():
    def __init__(self, mu, sd , results , index_results, df_ewma,cusum_np_arr
                 , list_1_2sD=[]
                 ,list_1_2sU=[], list_1_3sD=[],list_1_3sU=[], list_2_2sD=[]
                 ,list_2_2sU=[], list_R_4s=[],list_4_1sD=[], list_4_1sU=[]
                 ,list_10xD=[],list_10xU=[] ,list_8xU=[], list_8xD=[],list_12xD=[],list_12xU=[]
                 ,ewma_l_02U=[], ewma_l_02D=[],ewma_l_01U=[], ewma_l_01D=[]
                 ,ewma_l_005U=[], ewma_l_005D=[],Cp_l=[],Cn_l=[]):

        self.mu = mu
        self.sd = sd
        self.results = results
        self.index_results = index_results
        self.df_ewma = df_ewma
        self.cusum_np_arr = cusum_np_arr
        self.list_1_2sD = list_1_2sD
        self.list_1_2sU = list_1_2sU
        self.list_1_3sD = list_1_3sD
        self.list_1_3sU = list_1_3sU
        self.list_2_2sD = list_2_2sD
        self.list_2_2sU = list_2_2sU
        self.list_R_4s = list_R_4s
        self.list_4_1sD = list_4_1sD
        self.list_4_1sU = list_4_1sU
        self.list_10xD = list_10xD
        self.list_10xU = list_10xU
        self.list_8xU = list_8xU
        self.list_8xD = list_8xD
        self.list_12xD = list_12xD
        self.list_12xU = list_12xU
        self.ewma_l_02U = ewma_l_02U
        self.ewma_l_02D = ewma_l_02D
        self.ewma_l_01U = ewma_l_01U
        self.ewma_l_01D = ewma_l_01D
        self.ewma_l_005U = ewma_l_005U
        self.ewma_l_005D = ewma_l_005D
        self.Cp_l = Cp_l
        self.Cn_l = Cn_l


    def cusum_std(self):

        Cp=(cusum_np_arr*0).copy()
        Cm=Cp.copy()
        k=0.5
        h=5
        for ii in np.arange(len(cusum_np_arr)):
            if ii == 0:

                Cp[ii]=0
                Cm[ii]=0
            else:

                Cp[ii]=np.max([0,((cusum_np_arr[ii]-mu)/sd)-k+Cp[ii-1]])
                Cm[ii]=np.max([0,-k-((cusum_np_arr[ii]-mu)/sd)+Cm[ii-1]])
            Cont_limit_arr=np.array(h*np.ones((len(results),1)))
            Cont_lim_df= pd.DataFrame(Cont_limit_arr,columns = ["h"])
            cusum_df=pd.DataFrame({'Cp': Cp, 'Cn': Cm})
            cusum_df1=cusum_df.join(Cont_lim_df)
        for i in cusum_df1['Cp']:

            if i>=h:
                self.Cp_l.append(1)
            else:
                self.Cp_l.append(0)
        for j in cusum_df1['Cn']:
            if j>=h:
                self.Cn_l.append(1)
            else:
                self.Cn_l.append(0)
        return self.Cn_l, self.Cp_l


    def ewma_l_02(self): #bunu al
        λ = 0.2
        L = 2.962
        UCL_l=[]
        LCL_l=[]
        for ind in range (1,len(results)+1):
            UCL = mu + L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            LCL = mu - L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            ewm01= df_ewma.ewm(alpha=0.2).mean()
            UCL_l.append(UCL)
            LCL_l.append(LCL)
            df_UCL_l = pd.DataFrame(UCL_l, columns = ["UCL_l"])
            df_LCL_l = pd.DataFrame(LCL_l, columns = ["LCL_l"])

            df_ewmas= [ewm01,df_LCL_l, df_UCL_l]
            df = pd.concat(df_ewmas, axis=1)
        for i in range (len(results)):
            if (df.iloc[:,0][i] >= df.iloc[:,2][i]):
                self.ewma_l_02U.append(1)
            else:
                self.ewma_l_02U.append(0)
        for i in range (len(results)):
            if (df.iloc[:,0][i] <= df.iloc[:,1][i]):
                self.ewma_l_02D.append(1)
            else:
                self.ewma_l_02D.append(0)

        ewma_l_02U = np.array(self.ewma_l_02U)
        ewma_l_02D = np.array(self.ewma_l_02D)
        return ewma_l_02D, ewma_l_02U



    def ewma_l_01(self):
        λ = 0.1
        L = 2.814
        UCL_l=[]
        LCL_l=[]
        for ind in range (1,len(results)+1):
            UCL = mu + L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            LCL = mu - L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            ewm01= df_ewma.ewm(alpha=0.1).mean()
            UCL_l.append(UCL)
            LCL_l.append(LCL)
            df_UCL_l = pd.DataFrame(UCL_l, columns = ["UCL_l"])
            df_LCL_l = pd.DataFrame(LCL_l, columns = ["LCL_l"])

            df_ewmas= [ewm01,df_LCL_l, df_UCL_l]
            df = pd.concat(df_ewmas, axis=1)
        for i in range (len(results)):
            if (df.iloc[:,0][i] >= df.iloc[:,2][i]):
                self.ewma_l_01U.append(1)
            else:
                self.ewma_l_01U.append(0)
        for i in range (len(results)):
            if (df.iloc[:,0][i] <= df.iloc[:,1][i]):
                self.ewma_l_01D.append(1)
            else:
                self.ewma_l_01D.append(0)

        ewma_l_01U = np.array(self.ewma_l_01U)
        ewma_l_01D = np.array(self.ewma_l_01D)
        return ewma_l_01D,ewma_l_01U


    def ewma_l_005(self):
        λ = 0.05
        L = 2.615
        UCL_l=[]
        LCL_l=[]
        for ind in range (1,len(results)+1):
            UCL = mu + L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            LCL = mu - L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            ewm01= df_ewma.ewm(alpha=0.05).mean()
            UCL_l.append(UCL)
            LCL_l.append(LCL)
            df_UCL_l = pd.DataFrame(UCL_l, columns = ["UCL_l"])
            df_LCL_l = pd.DataFrame(LCL_l, columns = ["LCL_l"])

            df_ewmas= [ewm01,df_LCL_l, df_UCL_l]
            df = pd.concat(df_ewmas, axis=1)
        for i in range (len(results)):
            if (df.iloc[:,0][i] >= df.iloc[:,2][i]):
                self.ewma_l_005U.append(1)
            else:
                self.ewma_l_005U.append(0)
        for i in range (len(results)):
            if (df.iloc[:,0][i] <= df.iloc[:,1][i]):
                self.ewma_l_005D.append(1)
            else:
                self.ewma_l_005D.append(0)

        ewma_l_005U = np.array(self.ewma_l_005U)
        ewma_l_005D = np.array(self.ewma_l_005D)
        return ewma_l_005D,ewma_l_005U


    def westgard_1_2s(self):
        n = 2
        UCL = self.mu + n * self.sd
        LCL = self.mu - n * self.sd
        for i in results:
            if (i >= UCL):
                self.list_1_2sU.append(1)
            else:
                self.list_1_2sU.append(0)

        for i in results:
            if (i <= LCL):
                self.list_1_2sD.append(1)
            else:
                self.list_1_2sD.append(0)

        list_1_2sU = np.array(self.list_1_2sU)
        list_1_2sD = np.array(self.list_1_2sD)
        return list_1_2sD,list_1_2sU


    def westgard_1_3s(self):
        n = 3
        UCL = self.mu + n * self.sd
        LCL = self.mu - n * self.sd
        for i in results:
            if (i >= UCL):
                self.list_1_3sU.append(1)
            else:
                self.list_1_3sU.append(0)

        for i in results:
            if (i <= LCL):
                self.list_1_3sD.append(1)
            else:
                self.list_1_3sD.append(0)

        list_1_3sD = np.array(self.list_1_3sD)
        list_1_3sU = np.array(self.list_1_3sU)
        return list_1_3sD, list_1_3sU


    def westgard_2_2s(self):
        n = 2
        UCL = self.mu + n * self.sd
        LCL = self.mu - n * self.sd
        for j in range(0,len(results)-1):
            if (self.results[j] >= UCL and self.results[j+1] >= UCL):
                self.list_2_2sU.append(1)
            else:
                self.list_2_2sU.append(0)
        for j in range(0,len(results)-1):
            if (self.results[j] <= LCL and self.results[j+1] <= LCL):
                self.list_2_2sD.append(1)
            else:
                self.list_2_2sD.append(0)


        list_2_2sD = np.array(self.list_2_2sD)
        list_2_2sU = np.array(self.list_2_2sU)
        return list_2_2sD, list_2_2sU

    def westgard_R_4s(self):
        n = 2
        UCL = self.mu + n * self.sd
        LCL = self.mu - n * self.sd
        for j in range(0,len(results)-1):
            if (self.results[j] >= UCL and self.results[j + 1] <= LCL):
                self.list_R_4s.append(1)
            elif (self.results[j] <= LCL and self.results[j + 1] >= UCL):
                self.list_R_4s.append(1)
            else:
                self.list_R_4s.append(0)
        list_R_4s = np.array(self.list_R_4s)
        return list_R_4s

    def westgard_4_1s(self):
        n = 1
        UCL = self.mu + n * self.sd
        LCL = self.mu - n * self.sd
        for j in range(0,len(results)-3):
            if (self.results[j]>= UCL and self.results[j+1]>=UCL
                and self.results[j+2] >= UCL and self.results[j+3] >= UCL):
                self.list_4_1sU.append(1)
            else:
                self.list_4_1sU.append(0)
        for j in range(0,len(results)-3):
            if (self.results[j]<= LCL and self.results[j+1]<=LCL
                and self.results[j+2] <= LCL and self.results[j+3] <= LCL):
                self.list_4_1sD.append(1)
            else:
                self.list_4_1sD.append(0)
        list_4_1sD = np.array(self.list_4_1sD)
        list_4_1sU = np.array(self.list_4_1sU)
        return list_4_1sD, list_4_1sU

    def westgard_10x(self):

        for j in range(0,len(results)-9):
            if(self.results[j] > self.mu and self.results[j+1] > self.mu
               and self.results[j+2] > self.mu and self.results[j+3] > self.mu
               and self.results[j+4] > self.mu and self.results[j+5] > self.mu
               and self.results[j+6] > self.mu and self.results[j+7] > self.mu
               and self.results[j+8] > self.mu and self.results[j+9] > self.mu):
                self.list_10xU.append(1)
            else:
                self.list_10xU.append(0)
        for j in range(0,len(results)-9):
            if(self.results[j] < self.mu and self.results[j+1] < self.mu
               and self.results[j+2] < self.mu and self.results[j+3] < self.mu
               and self.results[j+4] < self.mu and self.results[j+5] < self.mu
               and self.results[j+6] < self.mu and self.results[j+7] < self.mu
               and self.results[j+8] < self.mu and self.results[j+9] < self.mu):
                self.list_10xD.append(1)
            else:
                self.list_10xD.append(0)
        list_10xD = np.array(self.list_10xD)
        list_10xU = np.array(self.list_10xU)
        return  list_10xD, list_10xU

    def westgard_8x(self):
        for j in range(0,len(results)-7):
            if(self.results[j] > self.mu and self.results[j+1] > self.mu
               and self.results[j+2] > self.mu and self.results[j+3] > self.mu
               and self.results[j+4] > self.mu and self.results[j+5] > self.mu
               and self.results[j+6] > self.mu and self.results[j+7]):
                self.list_8xU.append(1)
            else:
                        self.list_8xU.append(0)
        for j in range(0,len(results)-7):
            if(self.results[j] < self.mu and self.results[j+1] < self.mu
               and self.results[j+2] < self.mu and self.results[j+3] < self.mu
               and self.results[j+4] < self.mu and self.results[j+5] < self.mu
               and self.results[j+6] < self.mu and self.results[j+7]):
                self.list_8xD.append(1)
            else:
                self.list_8xD.append(0)
        list_8xD = np.array(self.list_8xD)
        list_8xU = np.array(self.list_8xU)
        return list_8xD, list_8xU

    def westgard_12x(self):
        for j in range(0,len(results)-11):
            if(self.results[j] > self.mu and self.results[j+1] > self.mu
               and self.results[j+2] > self.mu and self.results[j+3] > self.mu
               and self.results[j+4] > self.mu and self.results[j+5] > self.mu
               and self.results[j+6] > self.mu and self.results[j+7] > self.mu
               and self.results[j+8] > self.mu and self.results[j+9] > self.mu
               and self.results[j+10] > self.mu and self.results[j+11] > self.mu):
                self.list_12xU.append(1)
            else:
                self.list_12xU.append(0)
        for j in range(0,len(results)-9):
            if(self.results[j] < self.mu and self.results[j+1] < self.mu
               and self.results[j+2] < self.mu and self.results[j+3] < self.mu
               and self.results[j+4] < self.mu and self.results[j+5] < self.mu
               and self.results[j+6] < self.mu and self.results[j+7] < self.mu
               and self.results[j+8] < self.mu and self.results[j+9] < self.mu
               and self.results[j+10] < self.mu and self.results[j+11] < self.mu):
                self.list_12xD.append(1)
            else:
                self.list_12xD.append(0)
        list_12xD = np.array(self.list_12xD)
        list_12xU = np.array(self.list_12xU)
        return list_12xD, list_12xU


    def levey_jennings_graph(self):

        x = range(1,N+1)
        y = results

        plt.figure()
        plt.scatter(x, y , s= 0.5,c="b", alpha=0.8,label = "Result")


        UCL_1s = mu + 1 * sd
        LCL_1s = mu - 1 * sd
        UCL_2s = mu + 2 * sd
        LCL_2s = mu - 2 * sd
        UCL_3s = mu + 3 * sd
        LCL_3s = mu - 3 * sd

        plt.axhline(y=UCL_1s, xmin=0, xmax=N,c="r",ls="--",lw=0.25, label = "+1 sd" )
        plt.axhline(y=LCL_1s, xmin=0, xmax=N,c="r",ls="--",lw=0.25, label = "-1 sd")
        plt.axhline(y=UCL_2s, xmin=0, xmax=N,c="r",ls=":",lw=1,label = "+2 sd")
        plt.axhline(y=LCL_2s, xmin=0, xmax=N,c="r",ls=":",lw=1,label = "-2 sd")
        plt.axhline(y=UCL_3s, xmin=0, xmax=N,c="r",ls="-.",lw=1,label = "+3 sd")
        plt.axhline(y=LCL_3s, xmin=0, xmax=N,c="r",ls="-.",lw=1, label = "-3 sd")

        plt.axhline(y=mu, xmin=0, xmax=len(results),c="g",ls="--",lw=1)

        plt.legend(["+1 sd","-1 sd","+2 sd","-2 sd","+3 sd","-3 sd","mean"], loc="lower right", ncol=1,framealpha = 1)

        # Add titles
        plt.title("Levey Jennings plot", loc='left', fontsize=12, fontweight=0, color='red')
        plt.xlabel("Time")
        plt.ylabel("Result")

        plt.show()

    def ewma_graph_02(self):
        λ = 0.2
        L = 2.962
        UCL_l=[]
        LCL_l=[]
        for ind in range(1,len(results)+1):
            UCL = mu + L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            LCL = mu - L * sd*(((λ)*(1-(1- λ)**(2*ind))/(2- λ))**(0.5))
            ewm01= df_ewma.ewm(alpha=0.2).mean()
            UCL_l.append(UCL)
            LCL_l.append(LCL)
            df_UCL_l = pd.DataFrame(UCL_l, columns = ["UCL_l"])
            df_LCL_l = pd.DataFrame(LCL_l, columns = ["LCL_l"])


            x = range(1,len(results)+1)

            df_ewmas= [ewm01,df_LCL_l, df_UCL_l]
            df = pd.concat(df_ewmas, axis=1)
        # style
            plt.style.use('seaborn-colorblind')

        #multiple line plot
            plt.plot(x , df.iloc[:,0], marker='', color="b", linewidth=0.2, alpha=0.5, label ="EWMA values")
            plt.plot(x , df.iloc[:,1], marker='', color="r", linewidth=0.2, alpha=0.5, label = "LCL")
            plt.plot(x , df.iloc[:,2], marker='', color="r", linewidth=0.2, alpha=0.5, label = "UCL")
        # Add legend
            plt.legend(["ewma","LCL","UCL"], loc = "upper right", ncol=1, framealpha = 1)

        # Add titles
            plt.title("EWMA plot", loc='left', fontsize=12, fontweight=0, color='red')
            plt.xlabel("Time")
            plt.ylabel("Result")

        return plt.show()


    def cusum_std_graph(self):

        Cp=(cusum_np_arr*0).copy()
        Cm=Cp.copy()
        k=0.5
        h=5
        x = range(1,len(results)+1)
        for ii in np.arange(len(cusum_np_arr)):
            if ii == 0:

                Cp[ii]=0
                Cm[ii]=0
            else:
                Cp[ii]=np.max([0,((cusum_np_arr[ii]-mu)/sd)-k+Cp[ii-1]])
                Cm[ii]=np.max([0,-k-((cusum_np_arr[ii]-mu)/sd)+Cm[ii-1]])
        Cont_limit_arr=np.array(h*np.ones((len(results),1)))
        Cont_lim_df= pd.DataFrame(Cont_limit_arr,columns = ["h"])
        cusum_df=pd.DataFrame({'Cp': Cp, 'Cn': Cm})
        cusum_df1=cusum_df.join(Cont_lim_df)

        plt.style.use('seaborn-colorblind')

        #multiple line plot
        plt.plot(x , cusum_df1.iloc[:,0], marker='', color="b", linewidth=0.3, alpha=0.8, label ="Ci+")
        plt.plot(x , cusum_df1.iloc[:,1], marker='', color="g", linewidth=0.3, alpha=0.8, label = "Ci-")
        plt.plot(x , cusum_df1.iloc[:,2], marker='', color="r", linewidth=0.5, alpha=1, label = "h (CL)")
        # Add legend
        plt.legend(["Ci+","Ci-","h (CL)"], loc="upper right", ncol=1, framealpha = 1)

        # Add titles
        plt.title("CUSUM plot", loc="left", fontsize=12, fontweight=0, color='red')
        plt.xlabel("Time")
        plt.ylabel("Cusum Result")

        return plt.show()







"""  SİMÜLASYON VERİ ÜRETİMİ (M ÜRETİLMEK İSTENEN SAYI (HER DÜZEY İÇİN)) """
M = 1000 # istenen simülasyon veri sayısı bu!!!
# EKLENEN HATA SEVİYESİ SAYISI 31

"""hata ekleme simülasyon deneme"""
results_L1=[] # belirlenen k lara göre list of list
k_lists=[] # uygulanan k için list of lists

N = M # simülasyona giden adet

# mean and standard deviation
# albumin biologicalvariation.eu meta analysis CVI:2.6
# https://biologicalvariation.eu/search?q=albumin
# CVI: 2.6 => CVA sınır=1.3 , mean= 4.5 =>


""" Kullanıcı veri girişi """
# I <0.5* CVw
# B < 0.25*((CVw**2+CVg**2)**0.5)
# TE < 1.65*0.5*CVw+0.25*((CVw**2+CVg**2)**0.5)

mu=4.5
CVa=1.3

#mu = float(input ("enter target value or mean : ")) # ör: 4.5 albumin
#CVa = float(input ("enter analytical coefficient of variation : ")) # ör: 1.3 albumin
sd = mu*CVa/100
#CVı= float(input("enter within-subject biological variation (CVı) : ")) # ör: 2.6 albumin
#CVg= float(input("enter between-subject biological variation (CVg) : ")) # ör: 5.1 albumin
print("""

      Please wait

      """)
#I= 0.5*CVı # desirable imprecision
#B =0.25*((CVı**2+CVg**2)**0.5) # desirable bias
#TE = 1.65*0.5*CVı+0.25*((CVı**2+CVg**2)**0.5) # desirable specification for allowable total error







s = np.random.normal(mu, sd, N+11) # normal dağılımda N tane veri üretme


n_error = len(np.arange(0,3.5,2.5))
for k in np.arange(0,3.5,2.5):
    k_list = np.ones(N)*k
    k_lists.append(k_list)

    if k==0:   #sonradan 12x gibi kurallar için kesme yaptığımızdan ilk turda 11 fazla veri üretiyoruz
        results_t = list(s+sd*k)
        results_L1.append(results_t)
    else:
        results_t = list(s[-N:]+sd*k)
        results_L1.append(results_t)






results = [item for sublist in results_L1 for item in sublist] # üretilen sonuçlar: list of list to list dönüşümü

error_list = [item for sublist in k_lists for item in sublist] # eklenen hata: list of list to list dönüşümü

error_list_np = np.array(error_list)

"""bitti"""



index_results=list(range (1,len(results)+1))
# resultın indexini yaptık


df_ewma = pd.DataFrame(results, columns = ["ewma_data1"]) #ewma analizinde kullanılacak verinin dataframe'e dönüşümü
cusum_np_arr= np.array(results) # cusum analizinde kullanılacak verinin np array e dönüşümü






w = IQC_models(mu,sd,results,index_results,df_ewma, cusum_np_arr)

# Aşağıda son M(İSTENEN VERİ SAYISI)*31(HATA SEVİYESİ SAYISI) sonucu listeye çevirdim
Z = M*n_error
list_1_2sD,list_1_2sU = w.westgard_1_2s() #1 _2s düşük yüksek rule out kayıt
list_1_2sD = list_1_2sD[-Z:]
list_1_2sU = list_1_2sU[-Z:]

list_1_3sD, list_1_3sU = w.westgard_1_3s() # 1_3s düşük yüksek rule out kayıt
list_1_3sD = list_1_3sD[-Z:]
list_1_3sU = list_1_3sU[-Z:]

list_2_2sD, list_2_2sU = w.westgard_2_2s() # 2_2s düşük yüksek rule out kayıt
list_2_2sD = list_2_2sD[-Z:]
list_2_2sU = list_2_2sU[-Z:]

list_R_4s = w.westgard_R_4s() # R_4s düşük yüksek rule out kayıt
list_R_4s = list_R_4s[-Z:]

list_4_1sD, list_4_1sU = w.westgard_4_1s() # 4_1s düşük yüksek rule out kayıt
list_4_1sD = list_4_1sD[-Z:]
list_4_1sU = list_4_1sU[-Z:]

list_10xD, list_10xU = w.westgard_10x() # 10x düşük yüksek rule out kayıt
list_10xD = list_10xD[-Z:]
list_10xU = list_10xU[-Z:]

list_8xD, list_8xU = w.westgard_8x() # 8x düşük yüksek rule out kayıt
list_8xD = list_8xD[-Z:]
list_8xU = list_8xU[-Z:]

list_12xD, list_12xU = w.westgard_12x() # 8x düşük yüksek rule out kayıt
list_12xD = list_12xD[-Z:]
list_12xU = list_12xU[-Z:]

ewma_l_02D,ewma_l_02U = w.ewma_l_02() # ewma l=0.2 rule out kayıt
ewma_l_02D = ewma_l_02D[-Z:]
ewma_l_02U = ewma_l_02U[-Z:]

ewma_l_01D,ewma_l_01U = w.ewma_l_01() # ewma l=0.2 rule out kayıt
ewma_l_01D = ewma_l_01D[-Z:]
ewma_l_01U = ewma_l_01U[-Z:]

ewma_l_005D,ewma_l_005U = w.ewma_l_005() # ewma l=0.2 rule out kayıt
ewma_l_005D = ewma_l_005D[-Z:]
ewma_l_005U = ewma_l_005U[-Z:]

cusum_s_D,cusum_s_U = w.cusum_std() # ewma l=0.2 rule out kayıt
cusum_s_D = cusum_s_D[-Z:]
cusum_s_U = cusum_s_U[-Z:]

error_list_np = np.array(error_list)
error_list_np = error_list_np[-Z:]

# result ı da ekleyelim
results_list_np = np.array(results)

results_list_np = results_list_np[-Z:] # ilk 11 fazlaydı 12s nedeniyle

# tek dataframe oluşturma

df_list_1_2sU = pd.DataFrame(list_1_2sU, columns = ["1_2sU"])
df_list_1_2sD = pd.DataFrame(list_1_2sD, columns = ["1_2sD"])

df_list_1_3sU = pd.DataFrame(list_1_3sU,columns = ["1_3sU"])
df_list_1_3sD = pd.DataFrame(list_1_3sD,columns = ["1_3sD"])

df_list_2_2sU = pd.DataFrame(list_2_2sU,columns = ["2_2sU"])
df_list_2_2sD = pd.DataFrame(list_2_2sD,columns = ["2_2sD"])

df_list_R_4s = pd.DataFrame(list_R_4s,columns = ["R_4s"])

df_list_4_1sD = pd.DataFrame(list_4_1sD,columns = ["4_1sD"])
df_list_4_1sU = pd.DataFrame(list_4_1sU,columns = ["4_1sU"])

df_list_10xD = pd.DataFrame(list_10xD,columns = ["10xD"])
df_list_10xU = pd.DataFrame(list_10xU,columns = ["10xU"])

df_list_8xD = pd.DataFrame(list_8xD,columns = ["8xD"])
df_list_8xU = pd.DataFrame(list_8xU,columns = ["8xU"])

df_list_12xD = pd.DataFrame(list_12xD,columns = ["12xD"])
df_list_12xU = pd.DataFrame(list_12xU,columns = ["12xU"])

df_ewma_l_02D = pd.DataFrame(ewma_l_02D,columns = ["ewma_02_D"])
df_ewma_l_02U = pd.DataFrame(ewma_l_02U,columns = ["ewma_02_U"])

df_ewma_l_01D = pd.DataFrame(ewma_l_01D,columns = ["ewma_01_D"])
df_ewma_l_01U = pd.DataFrame(ewma_l_01U,columns = ["ewma_01_U"])

df_ewma_l_005D = pd.DataFrame(ewma_l_005D,columns = ["ewma_005_D"])
df_ewma_l_005U = pd.DataFrame(ewma_l_005U,columns = ["ewma_005_U"])

df_cusum_s_D = pd.DataFrame(cusum_s_D,columns = ["cusum_s_D"])
df_cusum_s_U = pd.DataFrame(cusum_s_U,columns = ["cusum_s_U"])


df_results = pd.DataFrame(results_list_np,columns = ["results"]) # results ı da ekledim

df_error = pd.DataFrame(error_list_np,columns = ["error"])


w_frames= [df_list_1_2sU,df_list_1_2sD, df_list_1_3sU,df_list_1_3sD,df_list_2_2sU
         ,df_list_2_2sD,df_list_R_4s,df_list_4_1sD,df_list_4_1sU,df_list_10xD
         ,df_list_10xU,df_list_8xD,df_list_8xU,df_list_12xD,df_list_12xU
         ,df_ewma_l_02D, df_ewma_l_02U, df_ewma_l_01D, df_ewma_l_01U
         ,df_ewma_l_005D, df_ewma_l_005U, df_cusum_s_D, df_cusum_s_U,df_results,df_error]


""" son toplu tablo"""
table_w_frames = pd.concat(w_frames, axis=1)


""" IQC graphs"""
# levey jennings graph

# w.levey_jennings_graph()

# ewma lambda=0.2 iken grafik

# w.ewma_graph_02()

# cusum standardized grafik

# w.cusum_std_graph()


""" ERROR STATUS """

conditions = [
    (table_w_frames['error']==0),
    (table_w_frames['error']> 0)
    ]

values = [0, 1]

table_w_frames['error_status'] = np.select(conditions, values)



""" MODEL LOAD """



import joblib

model = joblib.load(r'/Users/hikmetcancubukcu/Desktop/IQCAI_2/random_forest_IQCAI.joblib')



# data
 
veriler = table_w_frames

# if statement ile pozitif ve negatif hata seçimi eklenebilir.

selected_columns_train_v2 = ['1_2sU', '1_3sU',  '2_2sU',  'R_4s', 
       '4_1sU',  '10xU',  '8xU',  '12xU',
       'ewma_02_U',  'ewma_01_U',  'ewma_005_U',
        'cusum_s_U']

selected_columns_test_v2 = ['error_status'] # 0,1

""" ANN data preprocessing  """

X = veriler.loc[:,selected_columns_train_v2].values
Y = veriler.loc[:,selected_columns_test_v2].values



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_sc = sc.fit_transform(X)

y_pred0= model.predict(X_sc)
#y_pred= (y_pred0>0.508468) # youden
#y_pred1= y_pred.astype('uint8')



RF_pred=np.array(y_pred0)
df_RF_pred = pd.DataFrame(RF_pred,columns = ["RF_pred"])

son_w_frames= cusum_df1=table_w_frames.join(df_RF_pred)

""" ARL HESABI """

son_w_frames
arl_0=son_w_frames.tail(1000).idxmax()

arl_0= pd.DataFrame(arl_0-1000)
arl_0

count_alert=son_w_frames.tail(1000).sum(axis = 0, skipna = True)
count_alert
#export_excel= son_w_frames.to_excel(r"/Users/hikmetcancubukcu/Desktop/IQCAI_2/RF_model_outcome_0_4sd.xlsx", index= None, header= True)

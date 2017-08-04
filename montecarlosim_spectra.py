# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:20:51 2016

@author: Swopnil Ojha
"""
import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats

from numba import jit 

"""This code reads the database containing median spectra and simulates response 
spectra by using monte carlo simulation on the basis of the correlation coefficient 
between logarithmic spectral acceleration at two different periods as proposed by 
Jayaram and Baker

Journal Paper
Baker, J., & Jayaram, N. (2008). Correlation of spectral acceleration values from 
NGA ground motion models. Earthquake Spectra(24(1)), 299-317."""


class simulated_spectra():
    
    def __init__(self,file,n=20): #specify full path of the file
               
        self.n = n  #specify the number of ground motions to be generated
        filename = os.path.abspath(file)
        df = pd.read_table(filename, sep = ',',\
            skiprows = 0, usecols = ['F(Hz)','Median','LnStd'])
            
        self.w,self.median,self.LnStd = np.array(df['F(Hz)']), \
            np.array(df['Median']), np.array(df['LnStd'])
    

#correlation coefficient between logarithmic spectral acceleration at two different periods
#as proposed by Jayaram and Baker      
        
    def rho(self,T1,T2):
        Tmin = np.min([T1,T2])
        Tmax = np.max([T1,T2])
        c1 = 1 - np.cos(np.pi/2-0.366*np.log(Tmax/(np.max([Tmin,0.109]))))
        if Tmax < 0.2:
            c2 = 1-0.105*(1-1/(1+np.exp(100*Tmax-5)))*((Tmax-Tmin)/(Tmax-0.0099))
        else:
            c2 = 0
        if Tmax < 0.109:
            c3 = c2
        else:
            c3 = c1            
        c4 = c1 + 0.5*(np.sqrt(c3)-c3)*(1+np.cos(np.pi*Tmin/0.109))
        
        if Tmax < 0.109:
            return c2
        elif Tmin > 0.109:
            return c1
        elif Tmax < 0.2:
            return np.min([c2,c4])
        else:
            return c4
            
    
#unconditional coviariance matrix as proposed by Jayram and Baker        
    def covariance(self): 
        w,lnstd = self.w,self.LnStd
        T = (1/w)[::-1]
        lnstd = lnstd[::-1]
        lnstd2 = lnstd**2        
        #initialize a coviariance diagonal matrix with zero at non diagonal indices
        cov = np.diag(lnstd2)  
        #get the indices of upper triangular matrix and lower triangular matrix  
        ui1 = np.triu_indices(len(T),1)
        li1 = (ui1[1],ui1[0])
        #vectorize the correlation coefficient function 
        vfunc = np.vectorize(self.rho)
        Ti,Tj = T[ui1[0]],T[ui1[1]]
        lnstd_i,lnstd_j = lnstd[ui1[0]],lnstd[ui1[1]]
        cov[ui1] = vfunc(Ti,Tj)*lnstd_i*lnstd_j
        cov[li1] = cov[ui1]
        return cov
        
       
    def simulate(self): #method to generate a monte-carlo simulated spectra 
        w,med = self.w, self.median
        z = np.linalg.cholesky(self.covariance())             
        sim = np.exp(np.log(med)[::-1] + np.matmul(z,\
            np.random.normal(size = len(w))))[::-1]        
        return sim
    
        
    def groundmotions(self): #generate required number of ground motions 
        for groundmotions in range(self.n):
            yield self.simulate()
    
        
    #selection criteria method; data = a suite of ground motion records          
    def selection_criteria(self,data): 
        target_med = self.median
        weights = np.array([1,2])
        devMeanSim = np.mean(np.log(data),axis=0)-np.log(target_med)
        devSkewSim = stats.skew(np.log(data),axis=0)
        devSigSim = np.std(np.log(data),axis=0)-\
            np.sqrt(np.diagonal(self.covariance())[::-1])
        devTotalSim = weights[0]*np.sum(devMeanSim**2) + \
            weights[1]*np.sum(devSigSim**2)+ 0.1*(weights[0]+weights[1])*\
                np.sum(devSkewSim**2)        
        return np.min(np.abs(devTotalSim))
        
    
    # method to select the best suite out of n number of suites
    
    @jit         
    def best_suite(self,trials = 5):
        first_suite = np.array(list(self.groundmotions()))
        criteria = self.selection_criteria(first_suite)
        
        for suite in range(trials):
            next_suite = np.array(list(self.groundmotions()))
            next_criteria = self.selection_criteria(next_suite)
            
            if next_criteria < criteria:
                criteria = next_criteria
                bestsuite = next_suite
        return bestsuite
        
    #creating a database of the best matched simulated spectra 
    def database(self):
        df = pd.DataFrame(self.w)
        groundmotions = self.best_suite()
        df = pd.concat([df,pd.DataFrame(np.transpose(groundmotions))],axis = 1,\
                ignore_index = True)
        df.rename(columns={0:'F(Hz)'}, inplace = True)      
        #initialize the mean,standard deviation, mean_std and mean-std     
        lnmean = np.mean(np.log(df.values[:,1:]),axis = 1)
        median = np.exp(lnmean)        
        lnstd = np.std(np.log(df.values[:,1:]),axis = 1)           
        mean_up = np.exp(lnmean+lnstd)
        mean_down = np.exp(lnmean-lnstd)        
        #insert the mean and standard deviation inside the Dataframe     
        dic = {'LnStd':lnstd,'Mean-Std':mean_down,'Mean+Std':mean_up,'Median':median}
        for keys, values in zip(dic.keys(),dic.values()):
            df.insert(1,keys,values)
        return df 
        
    def write_to_file(self,output): #output = output folder
        outputfile = 'MonteCarlo_Simulated.csv'        
        outputpath = os.path.join(output,outputfile)
        df = self.database()        
        df.to_csv(outputpath)
        
    def plotting(self):
        df = self.database()
        fig = plt.figure('Response Spectrum of Monte-Carlo Simulations')
        ax = fig.add_subplot(111)
        ax.loglog(self.w,df.values[:,5:])
        ax.loglog(self.w,df['Median'],linewidth = 4, label = "$\mu$" + " spectrum")
        ax.loglog(self.w,df['Mean+Std'],'--',linewidth = 3, label = \
            "$\mu$"+ " + " + "$\sigma$ "+'spectrum')
        ax.loglog(self.w,df['Mean-Std'],'--',linewidth = 3, label = \
            "$\mu$"+ " - " + "$\sigma$ "+'spectrum')
        ax.set_xlabel('Frequency(Hz)',fontsize = 13)
        ax.set_ylabel('Psa(g)', fontsize = 13)
        ax.set_title('Monte-Carlo Simulated Response Spectra Ensemble',\
            fontsize = 15)        
        ax.legend(loc = 4)        
        plt.show()
    
        
def main():                
    path = "F:\Books\Thesis\python scripts\RealSpectra.csv"
    outputfolder = 'F:\Books\Thesis\python scripts'
    a = simulated_spectra(path)
    a.write_to_file(outputfolder)
    a.plotting()

if __name__== '__main__':
    main()

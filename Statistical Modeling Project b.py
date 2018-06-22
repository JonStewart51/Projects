# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 20:20:29 2015

@author: JonStewart51
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as smi
import scipy as sp
from statsmodels.formula.api import ols
from pandas.tools.plotting import scatter_matrix
import math

data1=pandas.read_csv("C:\Users\JonStewart51\Documents\Statistical Models Fall 2015\ch6prob9GroceryRetailer.csv", names=('y', 'x1', 'x2', 'x3'))
#f yeah

#scatterplot matrix seaborn
import seaborn as sns
#1)
A)

y = data1.y  # response
x1 = data1.x1  # predictor
x2 = data1.x2
x3 = data1.x3
data1['x4']=data1['x1']*data1['x2']
x4 = data1.x4
X=data[data.columns[1:4]]
data.head()

  #matrix format for data
""" plot scatterplot, look for patterns"""
"""Hav to match all dimensions, in 4-d plot"""

####OLS fit of data
#grocery_data=ols('y ~ x1 + x2 + C(x3)+ x4', data=data1).fit()




A)
plt.scatter(y,x1)
plt.xlabel("x1")
plt.ylabel("labor hours")

plt.plot(y)
plt.title("plot of y")

plt.scatter(y,x2)
plt.title("labor hours versus x2")

plt.scatter(y, x4)
plt.title("labor hours versus x4")


scatter_matrix(data1, alpha=0.2, figsize=(6, 6), diagonal='kde')


mlfull=ols('y ~ x1 + x2 + C(x3)+ x4', data=data1).fit()  #cond num=1.55e08  r2=.689 adjr2=.663
print mlfull.summary() #aic=669.6 r2=.689 ajdr2=.663

ml1=ols('y ~ x1 + C(x3)+ x4', data=data1).fit()
print ml1.summary()   #aic=667.8  r2=.688 r2adj= .668

ml2=ols('y ~ x2 + C(x3)+ x4', data=data1).fit()
print ml2.summary()  #aic=667.7  r2=.689 adjr2=.669
result=ml2   #based on aic and r2, this is the best model

ml3=ols('y ~ x1 + x2 + C(x3)', data=data1).fit()
print ml3.summary()   #aic 667.8  r2=.688 adjr2=.669

ml4=ols('y ~ C(x3)+ x4', data=data1).fit()
print ml4.summary() #aic=668.3 r2=.673 adjr2=.659

ml5=ols('y ~ x2 + C(x3)', data=data1).fit()
print ml5.summary() #aic 670.6 r2=.658 adjr2=.644



B)
#correlation matrix
np.corrcoef(data1)

 


#time plot versus the predictor variables

e0x1=x1[0:50]
e1x1=x1[1:51]
plt.scatter(e0x1,e1x1)
plt.xlabel("x1 e0")
plt.ylabel("x1 e1")

e0x2=x2[0:50]
e1x2=x2[1:51]
plt.scatter(e0x2, e1x2)
plt.xlabel("x2 e0")
plt.ylabel("x2 e1")

e0x3=x3[0:50]
e1x3=x3[1:51]
plt.scatter(e0x3,e1x3)
plt.xlabel("x3 e0")
plt.ylabel("x3 e0")

e0x4=x4[0:50]
e1x4=x4[1:51]
plt.scatter(e0x4,e1x4)
plt.xlabel("x4 e0")
plt.ylabel("x4 e1")


#plt.plot(x1,resid1)
#plt.plot(x2,resid1)
#plt.plot(x3, resid1)
#plt.plot(x4, resid1)



  #obtain same betahat as above.
res=result.resid     #89.
""" R^2"""
result.rsquared  

#residual plot over time
d=y-res

e1=d[0:51]
e2=d[1:52]
plt.scatter(e1,e2)  #there are clusters, due to the categorical variable
plt.xlabel("e1")
plt.ylabel("e2")
plt.title("residuals versus y")
###qq plot

smi.qqplot(res) #add in a line 
plt.title("QQ-plot")

##boxplot for residuals
plt.boxplot(res)
plt.title("boxplot of residuals")


#residuals versus y, x1, x2, x3, x4

plt.scatter(y, res)
plt.title("residuals versus y")

plt.scatter(x1, res)
plt.title("residuals versus X1")

plt.scatter(x2, res)
plt.title("residuals versus X3")

plt.scatter(x3, res)
plt.scatter(x4, res)
#Shapiro Wilks   are residuals normally distributed
sp.stats.shapiro(res)   #(0.9757532477378845, 0.36436957120895386) second value is pvalue, hence, fail to reject null of normality

#########Anova#############

#ols anova


#anova with only the class variable
grocery_anova=ols('y ~ x1 + x2 + C(x3) + x4', data=data1).fit()
table = smi.stats.anova_lm(grocery_anova) # Type 2 ANOVA DataFrame
print table  #p-value of 3.2987e-13
#print(grocery_anova.summary()) #aic=668.7   r2=.657, adjr2=.650  higher aic than lin reg

ml2=ols('y ~ x2 + C(x3)+ x4', data=data1).fit()
table2=smi.stats.anova_lm(ml2)
print table2

np.mean([data1.x3])
data1holiday=data1[data1.x3==1]
np.mean(data1holiday.y)  #4916.5
data0holiday=data1[data1.x3==0]
np.mean(data0holiday.y)   #4290.84   without holiday

############Part two#################

#generating a data set 
noise1=numpy.random.normal(0,50,60)
xsamp=sorted(numpy.random.uniform(0,50,60))
xsamp1=np.array(xsamp)
ysamp1=np.multiply(xsamp1,5)+noise1

# x versus x
e1_x1samp=xsamp1[0:59]
e2_x1samp=xsamp1[1:60]
plt.scatter(e1_x1samp, e2_x1samp)
plt.title("x time plot")


                   #*numpy.random.normal(0,2,50)
plt.plot(ysamp12)
result1 = sm.OLS( ysamp1, xsamp1 ).fit()
print result1.summary()  #obtain same betahat as above.
re1s= result1.mse_resid    #
#res1

#2) Vector of residuals
residualvector1=result1.resid    #vector of length 100

#3) sigma
res1= result1.mse_resid    
sigma_1=np.sqrt(res1)
sigma_1                #9.4632089938066084
#4  Diagnostics

#correlation in time
e1_q2p2=residualvector1[0:59]
e2_q2p2=residualvector1[1:60]
plt.scatter(e1_q2p2,e2_q2p2)
plt.title("residuals over time")
###qq plot
 #add in a line 
smi.qqplot(residualvector1)
plt.title("qq plot")
##boxplot for residuals
plt.boxplot(residualvector1)

#residuals versus y, xsamp1

plt.scatter(ysamp1, residualvector1)
plt.title("residual versus y")

plt.scatter(xsamp1, residualvector1)
plt.title("residuals versus x")

#shapiro wilks
sp.stats.shapiro(residualvector1)

##boxplot for residuals
plt.boxplot(residualvector1)
plt.title("Boxplot")


##second generating data set#################

noise2=numpy.random.normal(0,50,60)
x2samp=sorted(np.random.uniform(0,50,60))
x2samp1=numpy.array(xsamp)

functionsample=50+np.square(x2samp1) 
functionsample1=np.asarray(functionsample) 
functionsample1=np.multiply(functionsample1,.3 )   +noise2      
plt.plot(functionsample1)
plt.title("Generated Regression plot")

# x versus x
e1_x2samp=x2samp[0:59]
e2_x2samp=x2samp[1:60]
plt.scatter(e1_x2samp, e2_x2samp)
plt.title("x time plot")


result2 = sm.OLS( functionsample1, x2samp1 ).fit()
print result2.summary() 


#2) Vector of residuals

residualvector2=result2.resid    

#3) sigma
res2= result2.mse_resid    
sigma_2=np.sqrt(res2)
sigma_2
#4  Diagnostics

#correlation in time

d2=functionsample1-result2.fittedvalues
e1d2=d2[0:59]
e2d2=d2[1:60]
plt.scatter(e1d2, e2d2)   #same as below


e1_q2p3=residualvector2[0:59]
e2_q2p3=residualvector2[1:60]
plt.scatter(e1_q2p3, e2_q2p3)
plt.title("residuals versus time")


####clearly, this is not linear fit. Apply a transformation to the data, say take the sqrt
transformedy=np.sqrt(functionsample1)
transformedy1=np.asarray(transformedy)
plt.scatter(transformedy1,x2samp1)


result3= sm.OLS(transformedy1, x2samp1, missing="drop").fit() #59 observations
print result3.summary() 

#2) Vector of residuals

residualvector3=result3.resid    #vector of length 100

#3) sigma
res3= result3.mse_resid    
sigma_3=np.sqrt(res3)
sigma_3                #9.4632089938066084
#4  Diagnostics

#correlation in time
e1_q2p4=residualvector3[0:52]
e2_q2p4=residualvector3[1:53]
plt.scatter(e1_q2p4, e2_q2p4)
plt.title("residuals versus time")

###qq plot
smi.qqplot(residualvector3) #add in a line 
plt.title("qq plot")

##boxplot for residuals
plt.boxplot(residualvector3)

#residuals versus y, xsamp1

plt.scatter(transformedy[0:53], residualvector3[0:53])
plt.title("residual versus y")

plt.scatter(x2samp[0:53], residualvector3[0:53])
plt.title("residuals versus x")
#shapiro wilks
sp.stats.shapiro(residualvector3)  #data is not normal

##boxplot for residuals
plt.boxplot(residualvector3)
plt.title("boxplot")


























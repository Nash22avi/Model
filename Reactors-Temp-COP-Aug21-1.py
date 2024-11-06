#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd  
import numpy as np 
import os,sys
import subprocess
import glob
from os import path
import streamlit as st
import numpy as np
import joblib


# In[23]:


st.title("Active and Callibration Surface Temperature ")
st.markdown('**Objective** : Given the Core Temperature, Pressure, Absorbed Pressure and Mass of Palladium deposited, we predict the surface temperature of active reactor')
st.markdown('**Objective** : Given the Core Temperature and Pressure, we predict the surface temperature of Callibration Reactor and the COP')


# In[24]:


#Interface
st.markdown('Surface temperature predcition for Active and Callibration Reactor')
ACoreTemp = st.number_input('Input the active reactor core temperature', key = 1)
APressure = st.number_input('Input the active reactor pressure', key = 2)
AAbPressure = st.number_input('Input the active reactor Absorbed pressure', key = 3)
APdMass = st.number_input('Input the active reactor Palladium mass', format="%.5f", key = 4)


# In[25]:


def Active_temp_predict(ACoreTemp, APressure, AAbPressure, APdMass):
    import numpy as np

#    from sklearn.preprocessing import StandardScaler
#    objectA= StandardScaler()

#    scaled_XA = objectA.fit_transform(XA) 

    XmeanList = [6.36158333e+02, 3.00983333e+04, -3.88333333e+03, 2.30216667e-02]
#    print(XmeanList)
    XvarList = [1.11773091e+04, 4.65792881e+08, 1.14229222e+07, 3.44196306e-05]
#    print(XvarList)
#    XInputA = [497.5, 23790, -2160, 0.02146]
    XInputA = [ACoreTemp + 273, APressure, -AAbPressure, APdMass]
    XInputA = np.array(XInputA)
    XInputA = XInputA.reshape(1, -1)
#    scaled_XInputA = objectA.fit_transform(XInputA)

    for i in XInputA:
        for j in range(0, len(i)):
            XInputA[0][j] = (XInputA[0][j] - XmeanList[j])/(XvarList[j])**(1/2)

#    scaled_XA = pd.DataFrame(scaled_XA)

    best_degreeA = 1

    poly_bestA = PolynomialFeatures(degree= best_degreeA)
#    X_polyA = poly_bestA.fit_transform(scaled_XA)

#    X_polyA = pd.DataFrame(X_polyA)

    Active_Model = joblib.load('Active-Reactor-Temp-Predictor.joblib')

    XInputA = np.array(XInputA)
    XInputA = XInputA.reshape(1,-1)
    X_polyA = poly_bestA.fit_transform(XInputA)

    Active_SurfaceTemp = Active_Model.predict(X_polyA)
#    scaled_yA = objectA.fit_transform(yA)

    scaledyA_mean = 524.31666667
    scaledyA_std = 67.82720488
#    print(scaledyA_mean)
#    print(scaledyA_std)
    Active_SurfaceTemp = np.array(Active_SurfaceTemp)

    Active_SurfaceTempf = Active_SurfaceTemp*scaledyA_std + scaledyA_mean -273 
   # print(Active_SurfaceTempf[0][0])
    return Active_SurfaceTempf[0][0]


# In[28]:


#Active_temp_predict(497.5, 23790, 2160, 0.02146)


# In[20]:


if st.button("Predict the Active Surface temperature", key = 5): 
       st.success('The Active Reactor surface temperature is {}'.format(Active_temp_predict(ACoreTemp, APressure, AAbPressure, APdMass), format = "%.2f")) 


# In[42]:


def Callib_temp_predict(CalCoreTemp, CalPressure):    
#    datac = pd.read_excel("CAL EXP DATA1 (2).xlsx")

#    datafc = datac.dropna()

#    datafc.reset_index(inplace=True, drop=True)

#    datafc2 = datafc.drop(['Name'], axis = 1)

#    data_finalc = datafc2.loc[:,['Core','Pressure', 'Surface']]

#    Xc = data_finalc.iloc[0:,:2]
#    yc = data_finalc.iloc[:,-1]

#    yc = pd.DataFrame(yc)

#    for i in Xc:
#        if i == 'Core':
#            for j in range(0, len(Xc[i])):
#                Xc[i][j] = Xc[i][j] + 273

#    for i in yc:
#        for j in range(0, len(yc[i])):
#            yc[i][j] = yc[i][j] + 273

#    import seaborn as sns

#    from sklearn.preprocessing import StandardScaler
#    objectc= StandardScaler()

#    scaled_Xc = objectc.fit_transform(Xc) 
#    scaled_yc = objectc.fit_transform(yc) 

#    scaled_Xc = pd.DataFrame(scaled_Xc)
#    scaled_yc = pd.DataFrame(scaled_yc)

#    import matplotlib.pyplot as plt

#    scaled_yc = scaled_yc.loc[scaled_yc.index.intersection(scaled_Xc.index)]

#    from sklearn.preprocessing import StandardScaler
#    objectc= StandardScaler()

#    scaled_Xc = objectc.fit_transform(Xc) 
    XcmeanList = [666.19634921, 61743.46444444]
    XcvarList =  [2.94495540e+04, 1.49035833e+10]
    XInputc = [CalCoreTemp, CalPressure]
    XInputc = np.array(XInputc)
    XInputc = XInputc.reshape(1, -1)

#    scaled_XInputc = objectc.fit_transform(XInputc)

    best_degreec = 2

    for i in XInputc:    
        for j in range(0,len(i)):
            XInputc[0][j] = (XInputc[0][j] - XcmeanList[j])/(XcvarList[j])**(1/2)

    Callib_Model = joblib.load('Callib-Reactor-Temp-predictor.joblib')

    best_degreec = 2

    XInputc = np.array(XInputc)
    XInputc = XInputc.reshape(1,-1)
    poly_bestc = PolynomialFeatures(degree = best_degreec)
    X_polyc = poly_bestc.fit_transform(XInputc)

    Callib_SurfaceTempc = Callib_Model.predict(X_polyc)
#    scaled_yc = objectc.fit_transform(yc)
    scaledyc_mean = 561.41673492
    scaledyc_std = 161.90492471
    Callib_SurfaceTempfc = Callib_SurfaceTempc*scaledyc_std + scaledyc_mean -273
    return Callib_SurfaceTempfc[0]


# In[43]:


#Callib_temp_predict(412.1,1230)


# In[47]:


CalCoreTemp = st.number_input('Input the callibration reactor core temperature', key = 9)
CalPressure = st.number_input('Input the callibration reactor pressure', key = 10)


# In[44]:


if st.button("Predict the Callibration Surface temperature", key = 11): 
       st.success('The Callibration Reactor Surface Temperature is {}'.format(Callib_temp_predict(CalCoreTemp, CalPressure), format = "%.2f")) 


# In[45]:


def COP():
    cop = ((Active_temp_predict(ACoreTemp, APressure, AAbPressure, APdMass)+273)**4 - (Callib_temp_predict(CalCoreTemp, CalPressure) + 273)**4)/(Callib_temp_predict(CalCoreTemp, CalPressure) + 273)**4 + 1
    return cop


# In[46]:


if st.button("Predict the COP", key = 13): 
       st.success('The COP is {}'.format(COP()), format = "%.2f") 


# In[ ]:





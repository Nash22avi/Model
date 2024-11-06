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


# In[3]:


st.title("COP Calculator ")
st.markdown('**Objective** : Given the Core Temperature and Pressure, we calculate the surface temperature of Callibration Reactor and the COP')


# In[5]:


Active_SurfaceTemp = st.number_input('Input the active reactor Surface Temperature', format = "%.3f", key = 3)
Callibration_SurfaceTemp = st.number_input('Input the Callibration reactor surface temperature', format="%.3f", key = 4)


# In[1]:


def COP():
    cop = ((Active_SurfaceTemp + 273)**4 - (Callibration_SurfaceTemp + 273)**4)/(Callibration_SurfaceTemp + 273)**4 + 1
    return cop


# In[7]:


if st.button("Predict the COP", key = 13): 
       st.success('The COP is {}'.format(COP())) 


# In[ ]:





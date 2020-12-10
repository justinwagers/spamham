# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:18:55 2020

@author: machu
"""
import matplotlib.pyplot as plt
import pandas as pd 
data = pd.read_csv('spam.csv')
plt.bar(data["Category"].unique(), data['Category'].value_counts(), color = 'rgbkymc')
plt.title("Distribution of Labels")
plt.show()
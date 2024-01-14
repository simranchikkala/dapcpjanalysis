# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:55:39 2023

@author: simra
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
cpjd = pd.read_csv("C:\\Users\\simra\\OneDrive\\Desktop\\NM\\SEM III\\DAP\\cpj-database.csv", encoding_errors='ignore')
print(cpjd.info())
pd.set_option("display.max_columns" and "display.max_rows",None)
#Data Cleaning
cpjdata=cpjd.drop(axis=1, columns=["Unnamed: 17","Unnamed: 18","Unnamed: 19","Unnamed: 20","Unnamed: 21","Unnamed: 22","Unnamed: 23","Unnamed: 24","Unnamed: 25"])
fcpj=cpjdata.dropna(axis=0,subset=["Country Killed","Sex","Nationality","Organization","Job","Coverage"])
print(fcpj.info())

rv = {'Impunity (for Murder)': "Information NA", 'Taken Captive': 'Information NA', 'Threatened': "Information NA","Tortured": "Information NA"}
fcpj.fillna(value=rv, inplace=True)
condition= fcpj["Sex"]=="Sex"
ffcpj=fcpj[~condition]

print(ffcpj.info())
print("Dimensions of the CPJ Dataset: ",ffcpj.shape)
print("Size of the CPJ Dataset: ",ffcpj.size)

#Histogram plots 
plt.figure(figsize=(50, 6))
sns.histplot(ffcpj['Country Killed'],kde= True,color="lightgreen")
plt.title('Histogram on Country Killed')
plt.xlabel('Country Killed')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(50,6))
sns.histplot(ffcpj["Nationality"],kde= True)
plt.title("Histogram of Nationality of Killed Journalists")
plt.xlabel("Nationality")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


print(ffcpj.groupby("Sex").size())


from statsmodels.graphics.mosaicplot import mosaic

top_countries = ffcpj['Country Killed'].value_counts().sort_values(ascending=False).head(10).index

cpj_sub0 = ffcpj[ffcpj['Country Killed'].isin(top_countries)]
cpj_sub0.loc[:, 'Country Killed'] = pd.Categorical(cpj_sub0['Country Killed'])
print(cpj_sub0.groupby("Country Killed").size())

#Mosaic Plot
plt.figure(figsize=(12,6))
mosaic(cpj_sub0, ['Country Killed', 'Type of Death'], title='Country killed vs Type of death', gap=0.01, labelizer=lambda k: '', axes_label=True,label_rotation=45)
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=cpj_sub0["Country Killed"],kde= True,palette=['skyblue', 'salmon', 'green', 'purple', 'orange', 'cyan', 'pink', 'yellow', 'brown', 'gray',"lightgreen"],edgecolor="black")
plt.title("Top 10 Countries with Highest Journalists Killed")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

#Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose

ffcpj['Date'] = pd.to_datetime(ffcpj['Date'], errors='coerce')

result = seasonal_decompose(ffcpj['Date'], model='additive', period=1)
result.plot()
plt.show()

deaths_by_date = ffcpj.groupby('Date').size().reset_index(name='Number of Deaths')

deaths_by_date.set_index('Date', inplace=True)
plt.figure(figsize=(50, 6))
plt.plot(deaths_by_date.index, deaths_by_date['Number of Deaths'], marker='o', linestyle='-', color='orange')
plt.title('Number of Journalist Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Deaths')
plt.grid(True)
plt.show()

#Bar plots
plt.figure(figsize=(12, 6))
sns.barplot(x='Date', y='Number of Deaths', hue='Sex', data=ffcpj.groupby(['Date', 'Sex']).size().reset_index(name='Number of Deaths'), palette='viridis')
plt.title('Distribution of Journalist Deaths Over Time Based on Gender')
plt.xlabel('Date')
plt.ylabel('Number of Deaths')
plt.legend(title='Gender', loc='upper right')
plt.grid(True)
plt.show()


from datetime import datetime
topyear = ffcpj["Date"].dt.year.value_counts().sort_values(ascending=False).head(10).index
cpj_sub1 = ffcpj[ffcpj["Date"].dt.year.isin(topyear)]
print(cpj_sub0.groupby("Country Killed").size())
print(cpj_sub1.groupby(cpj_sub1["Date"].dt.year).size())
index=[1994,2004,2006,2007,2009,2012,2013,2014,2015,2017]

plt.figure(figsize=(10,6))
sns.histplot(data=cpj_sub1["Date"],kde= True)
plt.title("Years with Highest Journalists Killed")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks([1994,2004,2006,2007,2009,2012,2013,2014,2015,2017],rotation=90)
plt.show()


deathspd = cpj_sub1.groupby('Date').size().reset_index(name='Number of Deaths')
plt.figure(figsize=(50, 6))
plt.plot(deathspd.index, deathspd['Number of Deaths'], marker='o', linestyle='-', color='red')
plt.title('Number of Journalist Deaths per Year')
plt.xlabel('Date')
plt.ylabel('Number of Deaths')
plt.grid(True)
plt.show()
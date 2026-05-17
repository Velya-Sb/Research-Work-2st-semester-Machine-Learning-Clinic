#%matplotlib inline
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import Image
from matplotlib.pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import kernel_approximation
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
rcParams['figure.figsize'] = 15, 5
def features_plots(discrete_vars):
    plt.figure(figsize=(15,32))
    plt.subplots_adjust(hspace=0.7)
    for i, cv in enumerate(['Age']):
        plt.subplot(7, 2, i+1)
        plt.hist(data[cv], bins=len(data[cv].unique()))
        plt.title(cv)
        plt.ylabel('Frequency')
        plt.xlabel('')
        plt.xticks(rotation=0)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(6))
    for i, dv in enumerate(discrete_vars):
        plt.subplot(7, 2, i+3)
        data[dv].value_counts().plot(kind='bar', title=dv)
        plt.ylabel('Frequency')
        plt.xlabel('')
        plt.xticks(rotation=0)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(6))
    plt.show()
#data = pd.read_csv('C:\\Users\\User\\Desktop\\Research Works(NIR)\\MedicalCentre(Clinic)\\KaggleV2-May-2016.csv')
data = pd.read_csv('C:\\Users\\User\\Desktop\\Research Works(NIR)\\MedicalCentre(Clinic)\\KaggleV2-May-2016.csv')
print(data.head())
print(len(data))
# print(data[data['Age'] < 0]['Age'].value_counts().sum())
# data = data[data['Age'] >= 0]
# del data['Handcap']
# for field in ['Gender', 'No-show']:
#     data[field] = pd.Categorical(data[field]).codes
# data['AwaitingTime'] = data['AwaitingTime'].apply(lambda x: abs(x))
# dow_mapping = {'Monday' : 0, 'Tuesday' : 1, 'Wednesday' : 2, 'Thursday' : 3,
# 'Friday' : 4, 'Saturday' : 5, 'Sunday' : 6}
# data['DayOfTheWeek'] = data['DayOfTheWeek'].map(dow_mapping)
# print(len(data))

for column in list(data.columns):
    print("{0:25} {1}".format(column, data[column].nunique()))
discrete_vars = ['Gender', 'Scholarship', 'Hipertension', 
                 'Diabetes', 'Alcoholism', 
                 'SMS_received', 'No-show', 'Handcap']
features_plots(discrete_vars)

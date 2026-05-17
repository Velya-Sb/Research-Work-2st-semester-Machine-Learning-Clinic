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
    for i, cv in enumerate(['Age', 'Neighbourhood', 'AwaitingTime']):
        plt.subplot(7, 2, i+1)
        plt.hist(data[cv], bins=len(data[cv].unique()))
        plt.title(cv)
        plt.ylabel('Frequency')
        plt.xlabel('')
        plt.xticks(rotation=0)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(6))
    for i, dv in enumerate(discrete_vars):
        plt.subplot(7, 2, i+4)
        data[dv].value_counts().sort_index().plot(kind='bar', title=dv)
        plt.ylabel('Frequency')
        plt.xlabel('')
        plt.xticks(rotation=0)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(6))
    plt.show()
def week_in_year(row, col):
    month = row[f'{col}_month']
    day = row[f'{col}_day']
    monthes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    weekinyear = (sum(monthes[0:month - 1]) + day - 1)//7 + 1
    return weekinyear

def week_in_month(row, col):
    day = row[f'{col}_day']
    weekinmonth = (day - 1) // 7 + 1
    return weekinmonth

def weekendornot(row):
    day = row['Weekday']
    weekendornot = (day == 5) or (day == 6)
    return weekendornot

#data = pd.read_csv('C:\\Users\\User\\Desktop\\Research Works(NIR)\\MedicalCentre(Clinic)\\KaggleV2-May-2016.csv')
data = pd.read_csv('C:\\Users\\User\\Desktop\\Research Works(NIR)\\MedicalCentre(Clinic)\\KaggleV2-May-2016.csv')
print(data.head())
print(len(data))
print(data[data['Age'] < 0]['Age'].value_counts().sum())
data = data[data['Age'] >= 0]
del data['Handcap']
for field in ['Gender', 'No-show', 'Neighbourhood']:
    data[field] = pd.Categorical(data[field]).codes
del data['PatientId']
del data['AppointmentID']
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
data['AwaitingTime'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days + 1
print(data[data['AwaitingTime'] < 0]['AwaitingTime'].value_counts().sum())
data = data[data['AwaitingTime']>=0]
data['Weekday'] = data['AppointmentDay'].dt.dayofweek
print(data.head())
for col in ['ScheduledDay', 'AppointmentDay']:
    data[f'{col}_year'] = data[col].dt.year
    data[f'{col}_month'] = data[col].dt.month
    data[f'{col}_day'] = data[col].dt.day
for col in ['ScheduledDay', 'AppointmentDay']:
    data[f'{col}_hour'] = data[col].dt.hour
    data[f'{col}_minute'] = data[col].dt.minute
    data[f'{col}_second'] = data[col].dt.second
print(data.head())
for col in ['ScheduledDay', 'AppointmentDay']:
    data['%s_%s'%(col, 'NumWeekYear')] = data.apply(lambda row: week_in_year(row, col), axis = 1)
for col in ['ScheduledDay', 'AppointmentDay']:
    data['%s_%s'%(col, 'NumWeekMonth')] = data.apply(lambda row: week_in_month(row, col), axis = 1)
for col in ['ScheduledDay', 'AppointmentDay']:
    data['%s_%s'%(col, 'Weekend')] = data.apply(weekendornot, axis = 1)
print(data.head())
for column in list(data.columns):
    print("{0:25} {1}".format(column, data[column].nunique()))
discrete_vars = ['Gender', 'Scholarship', 'Hipertension', 
                 'Diabetes', 'Alcoholism', 
                 'SMS_received', 'No-show', 'Weekday']
features_plots(discrete_vars)
#Нужно проверить, является ли взаимосвязь между количеством людей с болезнями и их возрастом обратно пропорциональной в действительности или нет.
plt.scatter(data['Age'], data['AwaitingTime'], s=0.5)
plt.title('Scatter plot of Age and Awaiting Time')
plt.xlabel('Age')
plt.ylabel('Awaiting Time')
plt.xlim(0, 120)
plt.ylim(0, 120)
plt.show()
pd.set_option('display.width', 100)
pd.set_option('display.precision', 3)
correlations = data[['Age', 'AwaitingTime']].corr(method='pearson')
print(correlations)
#было интересно выяснить, увеличивает ли увеличение количества SMS-напоминаний вероятность явки пациента на прием. 
data_dow_status = data.groupby(['SMS_received', 'No-show'])['SMS_received'].count().unstack('No-show').fillna(0)
data_dow_status[[0, 1]].plot(kind='bar', stacked=True)
plt.title('Frequency of people showing up and not showing up by SMS receive')
plt.xlabel('Sms receive')
plt.ylabel('Frequency')
plt.show()

data_dow_status = data.groupby(['Weekday', 'No-show'])['Weekday'].count().unstack('No-show').fillna(0)
data_dow_status[[0, 1]].plot(kind='bar', stacked=True)
plt.title('Frequency of people showing up and not showing up by Day of the week')
plt.xlabel('Day of the week')
plt.ylabel('Frequency')
plt.show()
#понедельник - 0, воскресенье - 6

#a box plot of Age grouped by Status

data.boxplot(column=['Age'], return_type='axes', by='No-show')
plt.show()

#проанализировать зависимость возраста от статуса для обоих полов отдельно.

plt.figure(figsize=(15,3.5))
for i, status in enumerate(['show ups', 'no show ups']):
    data_show = data[data['No-show']==i]
    plt.subplot(1, 2, i+1)
    for gender in [0, 1]:
        data_gender = data_show[data_show['Gender']==gender]
        freq_age = data_gender['Age'].value_counts().sort_index()
        freq_age.plot()
    plt.title('Age wise frequency of patient %s for both genders'%status)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend(['Female', 'Male'], loc='upper left')

plt.show()

data.boxplot(column=['AwaitingTime'], return_type='axes', by='No-show')
plt.show()
#discrete_vars = ['Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received', 'No-show', 'Handcap']
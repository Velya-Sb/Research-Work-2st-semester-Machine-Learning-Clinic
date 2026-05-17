# Research-Work-2st-semester-Machine-Learning-Clinic

Код для печати всех диаграмм в первоначальном виде:
```python
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
```
Код для предварительной обработки данных и генерации признаков(Примечание: запускается несколько диаграмм. Чтобы перейти к следующей нужно закрыть текущую диаграмму)
```python
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
```
Код для использования разных моделей.(Аналогично предыдущему коду, модели запускаются последовательно, чтобы запустить следующую нужно закрыть оценку работу текущей, представленную в виде графика)
```python
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

def model_performance(model_name, X_train, y_train, y_test, Y_pred):
    print(f'Model name: {model_name}')
    print(f'Test accuracy (Accuracy Score):  {metrics.accuracy_score(y_test, Y_pred)}')
    print(f'Test accuracy (ROC AUC Score):  {metrics.roc_auc_score(y_test, Y_pred)}')
    print(f'Train accuracy:  {clf.score(X_train, y_train)}')
    fpr, tpr, thresholds = metrics.precision_recall_curve(y_test, Y_pred)
    print(f'Area Under the Precision-Recall Curve: {metrics.auc(fpr, tpr)}')
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, Y_pred)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
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
#features_plots(discrete_vars)
#Обеспечение перекрестной проверки путем разделения набора данных
features_of_choice = [u'Age', u'Gender', 'Weekday', 'Diabetes','Alcoholism', 'Hipertension', 'Scholarship','SMS_received',
'AwaitingTime', 'ScheduledDay_year','ScheduledDay_month','ScheduledDay_day', 'AppointmentDay_year', 'AppointmentDay_month',
'AppointmentDay_day', 'ScheduledDay_hour', 'ScheduledDay_minute', 'ScheduledDay_second', 
'ScheduledDay_NumWeekYear', 'ScheduledDay_NumWeekMonth', 'ScheduledDay_Weekend',
'AppointmentDay_NumWeekYear', 'AppointmentDay_NumWeekMonth', 'AppointmentDay_Weekend']
x = np.array(data[features_of_choice])
y = np.array(data['No-show'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)
# Классификация с помощью дерева решений
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
print(clf.get_params())
y_pred = clf.predict(x_test)
model_performance('Decision tree classifier', x_train, y_train, y_test, y_pred)

# Аппроксимация ядра
rbf_feature = kernel_approximation.RBFSampler(gamma=1, random_state=1)
X_train = rbf_feature.fit_transform(x_train)
clf = SGDClassifier()
clf.fit(X_train, y_train)
print(clf.get_params())
X_test = rbf_feature.fit_transform(x_test)
Y_pred = clf.predict(X_test)
model_performance('Kernel approximation', X_train, y_train, y_test, Y_pred)

#Классификация случайного леса

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
print(clf.get_params())
y_pred = clf.predict(x_test)
model_performance('Random Forest', x_train, y_train, y_test, y_pred)

#Градиентный бустинг

clf = GradientBoostingClassifier(random_state=10, learning_rate=0.1,n_estimators=200, max_depth=5, max_features=10)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

model_performance('Gradient Boosting', x_train, y_train, y_test, y_pred)
#Оценка важности

for feature, score in zip(features_of_choice, list(clf.feature_importances_)):
        print(f'({feature}, {score})')

#Градиентный бустинг(новый)
importances = clf.feature_importances_
important_features = np.where(importances > 0.01)[0]
x_train = x_train[:, important_features]
x_test = x_test[:, important_features]

clf = GradientBoostingClassifier(random_state=10, learning_rate=0.1, n_estimators=200, max_depth=5, max_features=min(10, len(important_features)))
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
model_performance('Gradient Boosting (important features only)', x_train, y_train, y_test, y_pred)
```

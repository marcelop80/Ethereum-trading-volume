#Regression Model - Ethereum Trade Volume

import kaggle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


kaggle.api.authenticate()


kaggle.api.dataset_download_files('varpit94/ethereum-data', path='.', unzip=True)

#Read
df = pd.read_csv('ETH-USD.csv')
df.head()
df.describe()
df.shape

#histogram
#df.hist(bins=20, figsize=(25,20), color='blue', alpha=0.8, edgecolor='black', linewidth=1.2)
#plt.show()

#Correlation matrix
#corr_matrix = df.drop(columns=['Date']).corr()
#plt.figure(figsize=(12, 10))
#sns.heatmap(corr_matrix, annot=True)
#plt.show()

#preprocessing
df_copy = df
#print(df_copy.isnull().sum())
#print(df_copy.duplicated().sum())
df_copy.drop_duplicates(inplace=True)

#print(df.dtypes)

#date do datetime
df['Date'] = pd.to_datetime(df['Date'])

#year and month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

#column YearMonth
df['YearMonth'] = df['Year'] * 100 + df['Month']
df['YearMonth'] = df['YearMonth'].astype(int)

#overwrite
df_copy['Date'] = df_copy['YearMonth']

#drop unnecessary columns
df_copy = df_copy.drop(['Year', 'Month', 'YearMonth'], axis=1)

#test
#print(df_copy.dtypes)
#print(df_copy.head())


#data split
x_ethereum = df_copy.iloc[:, :-1]
y_ethereum = df_copy.iloc[:, -1]

y_ethereum.shape
x_ethereum.shape 


EX_train, EX_test, Ey_train, Ey_test = train_test_split(
x_ethereum, y_ethereum, test_size = 0.3)

EX_train.shape
Ey_train.shape

#linear regression
Emodel = LinearRegression()
Emodel.fit(EX_train, Ey_train)

#predict
E_prediction = Emodel.predict(EX_test)
print('Predicted y for Ethereum', E_prediction)

#R2 score
E_r2 = r2_score(Ey_test, E_prediction)
print('R2 Score for Ethereum', E_r2)

#scatter plot

plt.scatter(Ey_test, E_prediction, color='blue', label= 'predicted vs atual')

#labels and title
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.title('comparasion of predicted and actual values')

#diagonal line for reference
plt.plot([Ey_test.min(), Ey_test.max()], [Ey_test.min(), Ey_test.max()], color='red', linestyle='--', label='ideal')

plt.legend()

#plt.show()

#minimum and maximum target values
min_target = Ey_test.min()
max_target = Ey_test.max()

print('Min target', min_target)
print('Max target', max_target)

#filter the rows for min and max target values
min_row = df_copy[df_copy['Volume'] == min_target]
max_row = df_copy[df_copy['Volume'] == max_target]

#print
print('Min row:')
print(min_row)
print('\nMax row:')
print(max_row)

#plotting volume change over time
plt.figure(figsize=(12, 6))
plt.plot(df_copy['Date'], df_copy['Volume'], color='blue')
plt.xlabel('Date') 
plt.ylabel('Volume')
plt.title('Volume over time')

plt.xticks(rotation=45) #for better readability

plt.tight_layout()#adjust the spacing


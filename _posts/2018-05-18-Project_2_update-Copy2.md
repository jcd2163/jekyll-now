
# DSI-US-4 Project 2 Regression Challenge
Predicting the price of homes at sale for the Aimes Iowa Housing dataset


```python
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet, LinearRegression, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```


```python
#read in datasets 
df = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
sample = pd.read_csv('./sample_sub_reg.csv')
```


```python
# explore training datasets
df.describe()
# df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>...</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2051.000000</td>
      <td>2.051000e+03</td>
      <td>2051.000000</td>
      <td>1721.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2029.000000</td>
      <td>...</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
      <td>2051.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1474.033642</td>
      <td>7.135900e+08</td>
      <td>57.008776</td>
      <td>69.055200</td>
      <td>10065.208191</td>
      <td>6.112140</td>
      <td>5.562165</td>
      <td>1971.708922</td>
      <td>1984.190151</td>
      <td>99.695909</td>
      <td>...</td>
      <td>93.833740</td>
      <td>47.556802</td>
      <td>22.571916</td>
      <td>2.591419</td>
      <td>16.511458</td>
      <td>2.397855</td>
      <td>51.574354</td>
      <td>6.219893</td>
      <td>2007.775719</td>
      <td>181469.701609</td>
    </tr>
    <tr>
      <th>std</th>
      <td>843.980841</td>
      <td>1.886918e+08</td>
      <td>42.824223</td>
      <td>23.260653</td>
      <td>6742.488909</td>
      <td>1.426271</td>
      <td>1.104497</td>
      <td>30.177889</td>
      <td>21.036250</td>
      <td>174.963129</td>
      <td>...</td>
      <td>128.549416</td>
      <td>66.747241</td>
      <td>59.845110</td>
      <td>25.229615</td>
      <td>57.374204</td>
      <td>37.782570</td>
      <td>573.393985</td>
      <td>2.744736</td>
      <td>1.312014</td>
      <td>79258.659352</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.263011e+08</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>12789.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>753.500000</td>
      <td>5.284581e+08</td>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7500.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.500000</td>
      <td>1964.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2007.000000</td>
      <td>129825.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1486.000000</td>
      <td>5.354532e+08</td>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9430.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1974.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>162500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2198.000000</td>
      <td>9.071801e+08</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11513.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>161.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2930.000000</td>
      <td>9.241520e+08</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>159000.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>...</td>
      <td>1424.000000</td>
      <td>547.000000</td>
      <td>432.000000</td>
      <td>508.000000</td>
      <td>490.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>611657.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 39 columns</p>
</div>




```python
# inspect features, data types, and missing values
# df.info()
# df.isnull().sum().sort_values(ascending=False)
```

Dataset contains 80 features and a target column of sale price. Some features have significant missing values or outliers which I will need to clean up. Below, I created a function to replace null values in Lot Frontage with a linear regression. I selected Lot Area and 1st Flr SF for my features in that regression. Initially, I got a low score and so I looked around online to see if anyone had encountered this before. I came across this website in my search: https://nycdatascience.com/blog/student-works/housing-prices-in-ames-iowa-kaggles-advanced-regression-competition/. They suggested taking the log of both the features and target for this exercise because that produced the most linear relationship. I updated my function with this suggestion and it returned a much higher score. I also visualized this below. In the first graphs before I took the logs, you can see that there is less of a linear relationship and presence of outliers. In the subsequent graphs, the outliers have been reduced and there is more of a linear relationship, especially between Lot Frontage and Lot Area. I will do something similar below with the target column as that distribution exhibits the qualities that warrant taking the log of the values.

Then I wrote another function to change the extreme outliers to reduce some affects later on with my scaling but decided against using it for this limited of a dataset. In the future, I would like to test the benefits of this function. 


```python
sns.pairplot(data=df, y_vars=['Lot Frontage'], x_vars=['Lot Area', '1st Flr SF'], size=10);
```


![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_6_0.png)


Lot Frontage vs Lot Area and 1st Flr SF. Not


```python
log_frontage = np.log(df['Lot Frontage'])
log_area = np.log(df['Lot Area'])
log_1st = np.log(df['1st Flr SF'])
```


```python
sns.jointplot(y=log_frontage, x=log_area, size=5);
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_9_1.png)



```python
sns.jointplot(y=log_frontage, x=log_1st, size=5);
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_10_1.png)



```python
# def fill_lot_frontage(dataframe):
#     mask = dataframe['Lot Frontage'].isnull() #create mask with null values of Lot Frontage
    
#     #split data based on mask
#     X1 = dataframe[~mask] 
#     X2 = dataframe[mask]
    
#     #instantiate linear regression based on two highest correlated variables: Lot Area and 1st Flr SF
#     lr = LinearRegression()
#     X_lot_train = X1.loc[:, ['Lot Area', '1st Flr SF']]
#     y_lot_train = X1['Lot Frontage']
   
#     #fit model 
#     lr.fit(X_lot_train,y_lot_train)
    
#     #replace null values in Lot Frontage with 0's
#     dataframe['Lot Frontage'].fillna(0, inplace=True)
    
#     #iterate over each row, check if Lot Frontage is 0, replace with 
#     for i in range(len(dataframe)):
#         if dataframe['Lot Frontage'][i] == 0:
#             dataframe['Lot Frontage'][i] = lr.intercept_ + lr.coef_[0] * dataframe['Lot Area'][i] + lr.coef_[1]* dataframe['1st Flr SF'][i]
        

# fill_lot_frontage(df)
# fill_lot_frontage(test)
```


```python
def fill_lot_frontage(dataframe):
    mask = dataframe['Lot Frontage'].isnull() #create mask with null values of Lot Frontage
    
    #split data based on mask
    X1 = dataframe[~mask] 
    X2 = dataframe[mask]
    
    #instantiate linear regression based on two highest correlated variables: Lot Area and 1st Flr SF
    lr = LinearRegression()
    X_lot_train = np.log(X1.loc[:, ['Lot Area', '1st Flr SF']])
    y_lot_train = np.log(X1['Lot Frontage'])
   
    #fit model 
    lr.fit(X_lot_train,y_lot_train)
    
    #replace null values in Lot Frontage with 0's
    dataframe['Lot Frontage'].fillna(0, inplace=True)
    
    #iterate over each row, check if Lot Frontage is 0, replace with 
    for i in range(len(dataframe)):
        if dataframe['Lot Frontage'][i] == 0:
            dataframe['Lot Frontage'][i] = np.exp(lr.intercept_ + lr.coef_[0] * np.log(dataframe['Lot Area'][i]) + lr.coef_[1]* np.log(dataframe['1st Flr SF'][i]))
        

fill_lot_frontage(df)
fill_lot_frontage(test)
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/ipykernel/__main__.py:22: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy


I originally tried to have this function not have to interate through each row and change all null values at once but that was giving me some issues for some reason. I would like to work on this again so that I can run this function after a TTS (since it iterates over each row, I need to run this before removing any outliers).


```python
assert df['Lot Frontage'].isnull().sum() == 0 & test['Lot Frontage'].isnull().sum()
```


```python
# # function to change upper outliers for float columns that represent area (SF)
# def change_outliers(dataframe):                           
#     for i in dataframe.columns: #iterate through columns
#         if dataframe[i].name not in ['Garage Yr Blt', 'Garage Cars', 'Baths', 'Bsmt Baths', 'SalePrice']: 
#             if dataframe[i].dtype == float: #iterate through columns that are floats and outside of the list above
#                 for j in range(1,len(dataframe)): #iterate through each row to detect and change outliers 
#                     if dataframe.loc[j,i] > dataframe[i].mean() + (6*dataframe[i].std()):
#                         dataframe.loc[j,i] = dataframe[i].mean() + (6*dataframe[i].std())


# change_outliers(df)
# change_outliers(test)
```


```python
#inspect float columns representing areas (note, I plotted a few iterations of this to get to this set of columns)
sns.pairplot(data=df, y_vars=['SalePrice'], x_vars=['Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 
                     'Total Bsmt SF', 'Gr Liv Area'],);
```


![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_16_0.png)


Clear outliers present above. Electing to remove them.


```python
#drop obvious outliers
df.drop(df[df['Total Bsmt SF']>4000].index, inplace=True)
df.drop(df[df['Lot Frontage']>300].index, inplace=True)
df.drop(df[df['Lot Area']>100000].index, inplace=True)
df.drop(df[df['Mas Vnr Area']>1500].index, inplace=True)
df.drop(df[df['Gr Liv Area']>4000].index, inplace=True)
df.drop(df[df['BsmtFin SF 1']>3000].index, inplace=True)
```


```python
#inspect updates
sns.pairplot(data=df, y_vars=['SalePrice'], x_vars=['Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 
                     'Total Bsmt SF', 'Gr Liv Area'],);
```


![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_19_0.png)


Updated distributions with outliers removed. Distributions now appear more linear, 0 values aside. 


```python
# set X as all columns except for SalePrice and y as target of SalePrice
X = df.iloc[:, :80]
y = df['SalePrice']
```


```python
# check for columns with null values 
# X.isnull().sum().sort_values(ascending=False).head(30)
```

Remaining null values are mostly in categorical features with datatype of objects. Will change these to 'NA' below


```python
# fill categorical values with 'NA' as per information on kaggle

def fill_categoricals(dataframe): #function takes in dataframe so that I can run on both my train and test set
    for i in dataframe.columns: #iterate through column of the dataframe
        if dataframe[i].dtype == object: #check if the datatype is object
            dataframe[i].fillna('NA', inplace=True) #fill null values with 'NA'

fill_categoricals(X)
fill_categoricals(test)
```

Lastly, I will fill the remaining null values:


```python
# for Garage Yr Blt, going to group by Year Built and take the mean. 
# since there will be some values as null after that,
# I am going to fill the remaining null value with the year built
# finally, going to fill the remaining null values with 0 

def fill_rest_nas(dataframe):
    dataframe['Garage Yr Blt'].fillna(dataframe.groupby(['Year Built'])['Garage Yr Blt'].transform('mean'), inplace=True)
    dataframe['Garage Yr Blt'].fillna(dataframe['Year Built'], inplace=True)
    dataframe.fillna(0, inplace=True)
    
fill_rest_nas(X)
fill_rest_nas(test)
```


```python
assert X.isnull().sum().sum() == 0 & test.isnull().sum().sum() == 0
```


```python
# I wanted to combine bathrooms for the house and basements
# Not sure if this affected anything but thought it was appropriate as this is usually how homes are sold

def combine_baths(dataframe):
    dataframe['Baths'] = dataframe['Full Bath'] + (dataframe['Half Bath']/2) #half baths listed as integer values
    dataframe['Bsmt Baths'] = dataframe['Bsmt Full Bath'] + (dataframe['Bsmt Half Bath']/2)
    dataframe.drop(['Full Bath', 'Half Bath','Bsmt Full Bath','Bsmt Half Bath'], axis=1, inplace=True) #drop columns after combining

combine_baths(X)
combine_baths(test)
```

Next, I created a function to dummy the categorical columns. For this, I needed to concatenate the training and prediction data, dummy the columns, and then split them up again. This was due to having different values in some of the features between datasets. 


```python
# function to dummy columns that are type object
# needed to combine training and predicition set, dummy the columns, and split them up again
def get_dummied(train, test):
    list = ['MS SubClass'] #MS SubClass is the only feature I want to dummy that is not an object 
    for i in train.columns:
        if train[i].dtype == object:
            list.append(i) #create list of columns to dummy
            
    full_data = pd.concat([train, test], axis=0) #concatenate training and prediction data 
    full_data = pd.get_dummies(full_data, columns=list, drop_first=True) #dummy the list of columns
    
    #split data back out 
    X_dummied = full_data[:len(train)] 
    test_dummied = full_data[len(train):]

    return X_dummied, test_dummied 

Xd, testd = get_dummied(X, test)
```


```python
assert Xd.shape[0] == X.shape[0] 
assert testd.shape[0] == test.shape[0]
assert Xd.shape[1] == testd.shape[1]
```

Now that my feature engineering is completed, I will move into modeling. Since I will be running elastic net cross validation below, I do not need to do any feature selection now. This will cause my model to take a long time to run, but will get me to the optimal set of features to use.

Before modeling I am going to take the log of the sale prices. As the distribution shows below, the majority of sale prices are located in the low end of the graph while higher prices constitute the tail of the distribution. By taking the log of the values, these qualities are reduced and the distribution becomes more normal. This will help in estimating the higher valued houses that would seem more like outliers without taking this approach. The second graph shows the updated distribution.

Later, I will have to take the exponential of the values after prediction.


```python
sns.distplot(df['SalePrice']);
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_33_1.png)



```python
sns.distplot(np.log(df['SalePrice']));
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_34_1.png)



```python
y = np.log(y) #take the log of sale prices

# create train and test sets from training data
X_train, X_test, y_train, y_test = train_test_split(Xd, y, test_size=0.2)
```


```python
# get interaction and polynomial features
poly = PolynomialFeatures()
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
```


```python
# use StandardScaler to standardize data
ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train_poly)
X_test_scaled = ss.fit_transform(X_test_poly)

```


```python
# # use ElasticNetCV to find optimal alpha and L1 ratio
# l1_ratios = np.linspace(0.01, 1.0, 25) #values to try for L1 ratio

# optimal_enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=100, cv=3, #let elastic net try 100 values for alpha
#                             selection='random') 
# optimal_enet.fit(X_train_scaled, y_train)

# print(optimal_enet.alpha_)
# print(optimal_enet.l1_ratio_)
```

Interestingly, after taking the log of the sale prices my L1 ratio went from 1 to .01. This is something I want to revisit to understand better. My thinking is that since the coefficients became much smaller due to the targets becoming much smaller, it was easier to completely zero them out. 


```python
# create ElasticNet model
enet = ElasticNet(alpha=0.35214938797030776, l1_ratio=0.01)
```


```python
# fit model to training data and score against test set from TTS
enet.fit(X_train_scaled, y_train)

y_hat_test = enet.predict(X_test_scaled)

#need to take exponential of both prediction and test to score
y_hat_test = np.exp(y_hat_test)

print(r2_score(np.exp(y_test), y_hat_test))
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    0.9489371639097156



```python
# plot predicted vs actuals
sns.jointplot(x=np.log(y_hat_test),y=y_test, kind='reg', size=12)
plt.title('Log Test Sale Price vs. Log Predicted Sale Price')
plt.ylabel('Log Test Sale Price')
plt.xlabel('Log Predicted Sale Price');
```

    /anaconda3/envs/dsi/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](/images/Project_2_update-Copy2_files/Project_2_update-Copy2_42_1.png)


Plot above showing Sale Prices from the test dataset of the TTS versus what the model predicted. 


```python
# run polynomial features and StandardScaler on prediction data 
test_poly = poly.fit_transform(testd)
test_scaled = ss.fit_transform(test_poly)
```


```python
# create SalePrice column of predicted values
test['SalePrice'] = np.exp(enet.predict(test_scaled)) #take exponential because predicting values based on log of sale prices
```


```python
# create new dataframe to submit
submit = test[['Id','SalePrice']]
submit.set_index('Id', inplace=True)
```


```python
#export submission data
submit.to_csv('./submit.csv')
```


```python
#model information
print('intercept: ', enet.intercept_)
print(df['SalePrice'].mean())

#code from lab to see percent zeroed out
print('Percent variables zeroed out:', np.sum((enet.coef_ == 0))/float(X_train_poly.shape[1]))

#print # of features 
print('# of features:', X_train_poly.shape[1])
print('# of features after regularization:', np.sum(enet.coef_>0))

#code from lab to create dataframe of sorted features and their coefficients
enet_coefs = pd.DataFrame({'features':poly.get_feature_names(Xd.columns),
                            'coef':enet.coef_,
                            'abs_coef':np.abs(enet.coef_)})

enet_coefs.sort_values('abs_coef', inplace=True, ascending=False)

enet_coefs.head(25)
```

    intercept:  12.026304719353643
    181315.65183374082
    Percent variables zeroed out: 0.9617960986382039
    # of features: 40755
    # of features after regularization: 711





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abs_coef</th>
      <th>coef</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1706</th>
      <td>0.008698</td>
      <td>0.008698</td>
      <td>Overall Cond Gr Liv Area</td>
    </tr>
    <tr>
      <th>13429</th>
      <td>0.006982</td>
      <td>-0.006982</td>
      <td>MS Zoning_C (all) Heating QC_TA</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>0.006829</td>
      <td>0.006829</td>
      <td>Year Built Gr Liv Area</td>
    </tr>
    <tr>
      <th>1416</th>
      <td>0.006703</td>
      <td>0.006703</td>
      <td>Overall Qual Overall Cond</td>
    </tr>
    <tr>
      <th>1427</th>
      <td>0.006700</td>
      <td>0.006700</td>
      <td>Overall Qual Gr Liv Area</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.006577</td>
      <td>0.006577</td>
      <td>Gr Liv Area</td>
    </tr>
    <tr>
      <th>9771</th>
      <td>0.006531</td>
      <td>-0.006531</td>
      <td>MS SubClass_30 Exter Cond_Fa</td>
    </tr>
    <tr>
      <th>2261</th>
      <td>0.006514</td>
      <td>0.006514</td>
      <td>Year Remod/Add Gr Liv Area</td>
    </tr>
    <tr>
      <th>4929</th>
      <td>0.006404</td>
      <td>0.006404</td>
      <td>Gr Liv Area Functional_Typ</td>
    </tr>
    <tr>
      <th>4749</th>
      <td>0.006333</td>
      <td>0.006333</td>
      <td>Gr Liv Area Street_Pave</td>
    </tr>
    <tr>
      <th>4714</th>
      <td>0.006268</td>
      <td>0.006268</td>
      <td>Gr Liv Area Garage Yr Blt</td>
    </tr>
    <tr>
      <th>1417</th>
      <td>0.006205</td>
      <td>0.006205</td>
      <td>Overall Qual Year Built</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>0.006165</td>
      <td>0.006165</td>
      <td>Overall Qual Year Remod/Add</td>
    </tr>
    <tr>
      <th>4725</th>
      <td>0.006142</td>
      <td>0.006142</td>
      <td>Gr Liv Area Yr Sold</td>
    </tr>
    <tr>
      <th>13409</th>
      <td>0.006127</td>
      <td>-0.006127</td>
      <td>MS Zoning_C (all) BsmtFin Type 1_BLQ</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>0.006041</td>
      <td>0.006041</td>
      <td>Year Built Year Remod/Add</td>
    </tr>
    <tr>
      <th>35876</th>
      <td>0.005998</td>
      <td>-0.005998</td>
      <td>Bsmt Qual_Fa Garage Cond_Fa</td>
    </tr>
    <tr>
      <th>1467</th>
      <td>0.005993</td>
      <td>0.005993</td>
      <td>Overall Qual Street_Pave</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.005988</td>
      <td>0.005988</td>
      <td>Overall Qual</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>0.005986</td>
      <td>0.005986</td>
      <td>Overall Qual Garage Yr Blt</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>0.005927</td>
      <td>0.005927</td>
      <td>Overall Qual Yr Sold</td>
    </tr>
    <tr>
      <th>35871</th>
      <td>0.005652</td>
      <td>-0.005652</td>
      <td>Bsmt Qual_Fa Garage Qual_Fa</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>0.005495</td>
      <td>0.005495</td>
      <td>Overall Cond Total Bsmt SF</td>
    </tr>
    <tr>
      <th>1393</th>
      <td>0.005436</td>
      <td>0.005436</td>
      <td>Lot Area Paved Drive_Y</td>
    </tr>
    <tr>
      <th>1415</th>
      <td>0.005115</td>
      <td>0.005115</td>
      <td>Overall Qual^2</td>
    </tr>
  </tbody>
</table>
</div>



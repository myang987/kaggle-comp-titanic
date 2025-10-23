# Introduction
This notebook is a personal exploration of an end-to-end data science process and serves as a learning process for myself as I continue to pursue a career in data science. <br>

The challenge is based on the Kaggle Titanic Competition that provides a dataset with different passenger attributes together with their survival status of the shipwreck. The aim is to develop a model capable of predicting passenger survival.



## Table of Contents 
Note: I can't get links to work 😡

[1. Exploratory Data Analysis](#explore) <br>
> [1.1 Preliminary observations](#prelim_explore) <br>
   [1.2 Exploring numerical attributes](#explore_num_columns) <br>
   [1.3 Exploring categorical attributes](#explore_cat_columns)  <br>
   [1.4 Univariate Analysis](#explore_cat_columns)  <br>
   [1.5 Bivariate Analysis](#explore_cat_columns)  <br>


### Imports


```python
# Core
import pandas as pd
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

### Reading Input


```python
# !unzip data/titanic.zip -d data/

test_data = pd.read_csv("data/test.csv")
train_data = pd.read_csv("data/train.csv")
pd.set_option("display.max_rows", None)
```

# 1. Exploratory Data Analysis
Objectives:
- Gain a preliminary understanding of available data
- Check for missing or null values
- Find potential outliers
- Assess correlations amongst attributes/features
- Check for data skew

[Back to contents](#top)

## 1.1 Preliminary observations
Examples from the dataset are shown below. <br><br>
[Back to contents](#top)


```python
print("train: ", train_data.shape)
train_data.head(5)
```

    train:  (891, 12)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("test: ", test_data.shape)
test_data.head(5)
```

    test:  (418, 11)





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Unique columns of train: ", set(train_data.columns) - set(test_data.columns))
print("Unique columns of test: ", set(test_data.columns) - set(train_data.columns))
```

    Unique columns of train:  {'Survived'}
    Unique columns of test:  set()



```python
print(train_data.dtypes)
print(test_data.dtypes)
```

    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object
    PassengerId      int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object


Initial observations show that the test dataset holds the same attributes as the train dataset with "Survived" ommitted. Test data is likely a subset of an original dataset with all passengers included. (This is infact the case stated in the competition explanation. I still checked to simulate a scenario where this is unknown)


```python
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB


### Numerical Columns from train_data


```python
# List of numerical features
num_features = train_data.select_dtypes(exclude='object').copy()
num_features.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], dtype='object')




```python
len(num_features.columns)
```




    7




```python
num_features.describe().round(decimals=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.00</td>
      <td>891.00</td>
      <td>891.00</td>
      <td>714.00</td>
      <td>891.00</td>
      <td>891.00</td>
      <td>891.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.00</td>
      <td>0.38</td>
      <td>2.31</td>
      <td>29.70</td>
      <td>0.52</td>
      <td>0.38</td>
      <td>32.20</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.35</td>
      <td>0.49</td>
      <td>0.84</td>
      <td>14.53</td>
      <td>1.10</td>
      <td>0.81</td>
      <td>49.69</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.42</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.50</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>20.12</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7.91</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>28.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>14.45</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.50</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>38.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>31.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>80.00</td>
      <td>8.00</td>
      <td>6.00</td>
      <td>512.33</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical Columns from train_data


```python
# List of categorical features
cat_features = train_data.select_dtypes(include='object').copy()
cat_features.columns
```




    Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')




```python
len(cat_features.columns)
```




    5




```python
cat_features.describe().round(decimals=2)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>347082</td>
      <td>B96 B98</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.isna().mean().sort_values(ascending=False)
```




    Cabin          0.771044
    Age            0.198653
    Embarked       0.002245
    PassengerId    0.000000
    Survived       0.000000
    Pclass         0.000000
    Name           0.000000
    Sex            0.000000
    SibSp          0.000000
    Parch          0.000000
    Ticket         0.000000
    Fare           0.000000
    dtype: float64



Notes for Data Cleaning & Processing:

3 features have missing values:
- Cabin
    - Likely drop this feature as 77% of the values are missing. This would have several adverse affects when training the model if I populated with synthetic information.
    - There is not enough organic data to provide the model with meaningful insights.
    - I may introduce skew into the final results.
    - Risk of overfitting with the 23% of real data.
- Age
    - Will populate the missing values. I feel this feature intuitively would be important for a person's survival rate. Younger = stronger and healthier. 
    - I could either use mean, median, or predict an age with other correlated features.
    - Using mean may introduce bias, variance shrinkage, and correlation distortion.
    - Using median is a better alternative as it limits the variance shrinkage, keeping the importance of higher variance data points.
    - Using a predictor such as KNN Imputation to give a best estimate would be the best solution.
- Embarked
    - This features signifies what port a passenger boarded the Titanic. Inherently, I don't feel this feature is that important due to the fact all passengers end up in the same shipwreck scenario. Although it could cause effects such as a person being more fatigued when boarding at a later port. 
    - I can either remove the rows with missing data, but I want to keep as much data of the other features as possible. The other solution is to populate the missing values with the most popular port of embarkment. Similar effects would occur as stated in the "Age" portion above, but given only 0.2% of data is missing, the effects would be minimal.

## 1.2 Univariate Analysis

[Back to contents](#top)

### Target Column
First, it is good practice to evaluate the skew of the target column as it may adversely affect the outcome of the prediction accuracy of regression models. This is not required (or possible) for our dataset as the target is a binary variable.

Note: Correcting skew is important for Linear Regression, but not necessary for Decision Trees and Random Forests.


```python
plt.figure()
sns.histplot(
    train_data.Survived, kde=True,
    stat="percent", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4)
)
plt.title('Distribution of Survived')
plt.show()
```


    
![png](README_files/README_25_0.png)
    


### Numerical Features


```python
fig = plt.figure(figsize=(12,18))
for i in range(len(num_features.columns)):
    fig.add_subplot(9,4,i+1)
    sns.histplot(
    num_features.iloc[:,i].dropna(), kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4)
)
    plt.xlabel(num_features.columns[i])

plt.tight_layout(pad=1.0)
```


    
![png](README_files/README_27_0.png)
    



```python
fig = plt.figure(figsize=(12, 18))

for i in range(len(num_features.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=num_features.iloc[:,i])

plt.tight_layout()
```


    
![png](README_files/README_28_0.png)
    



```python
train_data.loc[[train_data['SibSp'].idxmax(), train_data['Fare'].idxmax()]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>160</td>
      <td>0</td>
      <td>3</td>
      <td>Sage, Master. Thomas Henry</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>CA. 2343</td>
      <td>69.5500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



Notes for Data Cleaning & Processing:

There are two noticable outliers in the SibSp and Fare features. 
- Thomas Henry Sage has 8 siblings on board.
- Anna Ward paid an extraordinary amount of money ($512) for her ticket.

I can train the model multiple times trying different methods to improve the accuracy:
- Do nothing, train on original data.
- Remove the rows.
- Log transform the features given they are right skewed.


### Categorical Features


```python
cat_features.columns
```




    Index(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], dtype='object')




```python
cat_features_visual = cat_features[['Sex', 'Cabin', 'Embarked']]

fig = plt.figure(figsize=(18,20))
for index in range(len(cat_features_visual.columns)):
    plt.subplot(9,5,index+1)
    sns.countplot(x=cat_features_visual.iloc[:,index], data=cat_features_visual.dropna())
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
```


    
![png](README_files/README_33_0.png)
    



```python
fig = plt.figure(figsize=(18,20))
for index in range(len(cat_features_visual.columns)):
    plt.subplot(9,5,index+1)
    sns.boxplot(y=train_data.Survived, x=cat_features_visual.iloc[:,index])
    plt.xticks(rotation=90)
fig.tight_layout(pad=1.0)
```


    
![png](README_files/README_34_0.png)
    



```python

```

Notes for Data Cleaning & Processing:

Looking at the categorical variables, none of the features heavily skew out of the norm.

Sex seems to be a very important feature on the survival rate of the passenger.

## 1.3 Bivariate Analysis

[Back to contents](#top)

### Correlation Matrix


```python
plt.figure(figsize=(7,6))
plt.title('Correlation of numerical attributes', size=12)
correlation = num_features.corr()
sns.heatmap(correlation)
```




    <Axes: title={'center': 'Correlation of numerical attributes'}>




    
![png](README_files/README_39_1.png)
    


Upon first look:
- Fare and PClass are heavily inversely related. This makes sense because lower PClass value means better class, equating to more expensive fare.
- There is a moderate inverse relationship between Survived and PClass, signifying the better class the passenger is staying in, the higher the survival rate.

One consideration for the data cleaning process is the removal of the "Fare" feature. This is because of the heavy implied relationship between fare and PClass. The two features are correlated and impacts the outcome of "Survived". Having both features in training could over-fit for the PClass/Fare which signify the same thing. Ie. Multicollinearity

### Correlation between numeric features and target


```python
correlation = train_data.select_dtypes(exclude=['object']).corr()
correlation[['Survived']].sort_values(['Survived'], ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0.081629</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>-0.005007</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.035322</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.077221</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.338481</td>
    </tr>
  </tbody>
</table>
</div>



### Scatterplot between numeric features and target


```python
fig = plt.figure(figsize=(20,20))
for index in range(len(num_features.columns)):
    plt.subplot(10,5,index+1)
    sns.scatterplot(x=num_features.iloc[:,index], y=train_data.Survived, data=num_features.dropna())
fig.tight_layout(pad=1.0)
```


    
![png](README_files/README_44_0.png)
    





```python
sns.pairplot(train_data, hue="Survived")
```




    <seaborn.axisgrid.PairGrid at 0x147dd4ad0>




    
![png](README_files/README_46_1.png)
    



```python

```

# 2. Data Cleaning

Steps I'll take for pre-processing the data for training:
1. Removing redundant features
2. Dealing with outliers
3. Filling in missing data

## 2.1 Removing redundant features

From the correlation matrix, I've identified the following features to be highly correlated:
- Fare and PClass

Removing correlated feature to reduce multicollinearity

Also don't need identifiers columns PassengerId, Ticket, and Name


```python
train_data_copy = train_data.copy()

train_data_copy.drop(['Fare', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
```

Also removing Cabin column for having too many missing values as stated in section 1.1


```python
train_data_copy.drop(['Cabin'], axis=1, inplace=True)
train_data_copy.columns
```




    Index(['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked'], dtype='object')



## 2.2 Dealing with outliers

From before, we noticed 2 outlier points occuring in Fare and SibSp. Since we have remove the Fare feature due to multicollinearity, we only have to deal with the outlier in SibSp.


```python
fig = plt.figure(figsize=(20,5))
for index,col in enumerate(train_data_copy.columns):
    plt.subplot(1,7,index+1)
    sns.boxplot(y=col, data=train_data_copy)
fig.tight_layout(pad=1.5)
```


    
![png](README_files/README_54_0.png)
    



```python
train_data_copy[train_data_copy['SibSp'] >= 8]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>201</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>324</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>792</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>846</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
    <tr>
      <th>863</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data_copy = train_data_copy.drop(train_data_copy[train_data_copy['SibSp'] >= 8].index)
```


```python
train_data_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 884 entries, 0 to 890
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  884 non-null    int64  
     1   Pclass    884 non-null    int64  
     2   Sex       884 non-null    object 
     3   Age       714 non-null    float64
     4   SibSp     884 non-null    int64  
     5   Parch     884 non-null    int64  
     6   Embarked  882 non-null    object 
    dtypes: float64(1), int64(4), object(2)
    memory usage: 55.2+ KB


## 2.3 Filling in missing values


```python
pd.DataFrame(train_data_copy.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>170</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Numerical Features


```python
print("Mean of Age: ", train_data_copy['Age'].mean())
print("Median of Age: ", train_data_copy['Age'].median())
```

    Mean of Age:  29.69911764705882
    Median of Age:  28.0


As stated in the univariate analysis section, we will be using the median value as the simple method of filling NaNs.


```python
train_data_copy['Age'].fillna(train_data_copy['Age'].median(), inplace=True)
print("Mean of Age: ", train_data_copy['Age'].mean())
print("Median of Age: ", train_data_copy['Age'].median())
```

    Mean of Age:  29.372364253393663
    Median of Age:  28.0


The median value has stayed the same while the mean has decreased a little. This is something we have to deal with as the feature is important for training.

### Categorical Features

We will populate the two missing values of Embarked with the most common value.


```python
train_data_copy['Embarked'].value_counts()
```




    Embarked
    S    637
    C    168
    Q     77
    Name: count, dtype: int64




```python
train_data_copy['Embarked'].fillna("S", inplace=True)
train_data_copy['Embarked'].value_counts()
```




    Embarked
    S    639
    C    168
    Q     77
    Name: count, dtype: int64




```python
train_data_copy.info()
features = train_data_copy.drop('Survived', axis=1).columns.tolist()
features
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 884 entries, 0 to 890
    Data columns (total 7 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  884 non-null    int64  
     1   Pclass    884 non-null    int64  
     2   Sex       884 non-null    object 
     3   Age       884 non-null    float64
     4   SibSp     884 non-null    int64  
     5   Parch     884 non-null    int64  
     6   Embarked  884 non-null    object 
    dtypes: float64(1), int64(4), object(2)
    memory usage: 55.2+ KB





    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']




```python
train_data_copy = pd.get_dummies(train_data_copy)
```


```python
train_data_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 884 entries, 0 to 890
    Data columns (total 10 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   Survived    884 non-null    int64  
     1   Pclass      884 non-null    int64  
     2   Age         884 non-null    float64
     3   SibSp       884 non-null    int64  
     4   Parch       884 non-null    int64  
     5   Sex_female  884 non-null    bool   
     6   Sex_male    884 non-null    bool   
     7   Embarked_C  884 non-null    bool   
     8   Embarked_Q  884 non-null    bool   
     9   Embarked_S  884 non-null    bool   
    dtypes: bool(5), float64(1), int64(4)
    memory usage: 45.8 KB


# 4. Modeling


```python
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier
```


```python
# Series to collate mean absolute errors for each algorithm
mae_compare = pd.Series()
mae_compare.index.name = 'Algorithm'

train_X = train_data_copy.drop('Survived', axis=1)
train_y = train_data_copy[['Survived']]

X_test = pd.get_dummies(test_data[features])

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

rf_model.fit(train_X, train_y)
predictions = rf_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

```

    Your submission was successfully saved!


    /Users/mikeyang/Code/Notebooks/.venv/lib/python3.14/site-packages/sklearn/base.py:1365: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)



```python

```


```python

```

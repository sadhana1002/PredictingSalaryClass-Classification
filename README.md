
# Classification - Predict Salary group 



## Dataset

https://archive.ics.uci.edu/ml/datasets/Adult

Abstract: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

## Attribute Information:

## Listing of attributes: 

Labels : >50K, <=50K. 

age: continuous. 
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
fnlwgt: continuous. 
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
education-num: continuous. 
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
sex: Female, Male. 
capital-gain: continuous. 
capital-loss: continuous. 
hours-per-week: continuous. 
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Scope of this notebook

In this notebook, various classification algorithms are fed the training data (part of entire set) and the scores are compared. 
Just as a learning mechanism & to confirm how different algorithms work with adults dataset


```python
import pandas as pd
import matplotlib.pyplot as plt
```


```python
adults = pd.read_csv('adult.csv',names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])
adults_test = pd.read_csv('adult.csv',names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])

```


```python
train_data = adults.drop('label',axis=1)

test_data = adults_test.drop('label',axis=1)

data = train_data.append(test_data)

label = adults['label'].append(adults_test['label'])
```


```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_dataset = adults.append(adults_test)
```


```python
label.head()
```




    0     <=50K
    1     <=50K
    2     <=50K
    3     <=50K
    4     <=50K
    Name: label, dtype: object




```python
data_binary = pd.get_dummies(data)

data_binary.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>fnlwgt</th>
      <th>education_num</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>workclass_ ?</th>
      <th>workclass_ Federal-gov</th>
      <th>workclass_ Local-gov</th>
      <th>workclass_ Never-worked</th>
      <th>...</th>
      <th>native_country_ Portugal</th>
      <th>native_country_ Puerto-Rico</th>
      <th>native_country_ Scotland</th>
      <th>native_country_ South</th>
      <th>native_country_ Taiwan</th>
      <th>native_country_ Thailand</th>
      <th>native_country_ Trinadad&amp;Tobago</th>
      <th>native_country_ United-States</th>
      <th>native_country_ Vietnam</th>
      <th>native_country_ Yugoslavia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>77516</td>
      <td>13</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>83311</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>215646</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>234721</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>338409</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 108 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_binary,label)
```


```python
performance = []
```


```python
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
```


```python
 # Binary data
GNB.fit(x_train,y_train)
train_score = GNB.score(x_train,y_train)
test_score = GNB.score(x_test,y_test)
print(f'Gaussian Naive Bayes : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'Gaussian Naive Bayes', 'training_score':train_score, 'testing_score':test_score})
```

    Gaussian Naive Bayes : Training score - 0.7961753444851661 - Test score - 0.7928259934893435
    


```python
# LogisticRegression
from sklearn.linear_model import LogisticRegression


logClassifier = LogisticRegression()
```


```python
logClassifier.fit(x_train,y_train)
train_score = logClassifier.score(x_train,y_train)
test_score = logClassifier.score(x_test,y_test)

print(f'LogisticRegression : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'LogisticRegression', 'training_score':train_score, 'testing_score':test_score})
```

    LogisticRegression : Training score - 0.7986527712372802 - Test score - 0.7952214237454702
    


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
knn_scores = []


```


```python
train_scores = []
test_scores = []

for n in range(1,20,2):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    train_score = knn.score(x_train,y_train)
    test_score = knn.score(x_test,y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'KNN : Training score - {train_score} -- Test score - {test_score}')
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score})
    
plt.scatter(x=range(1, 20, 2),y=train_scores,c='b')
plt.scatter(x=range(1, 20, 2),y=test_scores,c='r')

plt.show()
```

    KNN : Training score - 0.9999795253987429 -- Test score - 0.9323751612308826
    KNN : Training score - 0.946233697098749 -- Test score - 0.7712671211842025
    KNN : Training score - 0.8647652586965869 -- Test score - 0.8119894355383576
    KNN : Training score - 0.847730390450646 -- Test score - 0.7886493458632762
    KNN : Training score - 0.8347085440511046 -- Test score - 0.7997051778146306
    KNN : Training score - 0.8288528080915624 -- Test score - 0.7950371598796143
    KNN : Training score - 0.8205196453799062 -- Test score - 0.7985381733308765
    KNN : Training score - 0.8186769312667636 -- Test score - 0.7991523862170629
    KNN : Training score - 0.815093876046764 -- Test score - 0.7985995946194951
    KNN : Training score - 0.8123502794783072 -- Test score - 0.7995823352373933
    


![png](output_19_1.png)



```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

knn.score(x_train,y_train)

train_score = knn.score(x_train,y_train)
test_score = knn.score(x_test,y_test)

print(f'K Neighbors : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'K Neighbors', 'training_score':train_score, 'testing_score':test_score})
```

    K Neighbors : Training score - 0.8647652586965869 - Test score - 0.8119894355383576
    


```python

```




    [{'algorithm': 'Gaussian Naive Bayes',
      'testing_score': 0.79282599348934346,
      'training_score': 0.79617534448516614},
     {'algorithm': 'LogisticRegression',
      'testing_score': 0.79522142374547022,
      'training_score': 0.79865277123728018},
     {'algorithm': 'K Neighbors',
      'testing_score': 0.81198943553835756,
      'training_score': 0.86476525869658694}]




```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rndTree = RandomForestClassifier()
```


```python
rndTree.fit(x_train,y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
rndTree.score(x_test,y_test)
```




    0.94846753884896506




```python
rndTree.score(x_train,y_train)
```




    0.99608935115988617




```python
train_score = rndTree.score(x_train,y_train)
test_score = rndTree.score(x_test,y_test)

print(f'Random Forests : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'Random Forests', 'training_score':train_score, 'testing_score':test_score})
```

    Random Forests : Training score - 0.9960893511598862 - Test score - 0.9484675388489651
    


```python
from sklearn import svm

svc = svm.SVC(kernel='linear')


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(data_binary,label)


```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python
x_train_scaled = scaler.transform(x_train)
```


```python
x_test_scaled = scaler.transform(x_test)
```


```python
svc.fit(x_train_scaled,y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
svc.score(x_test_scaled,y_test)
```




    0.85013205577053008




```python
train_score = svc.score(x_train_scaled,y_train)
test_score = svc.score(x_test_scaled,y_test)

print(f'Support Vector Machine: Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'Support Vector Machine', 'training_score':train_score, 'testing_score':test_score})
```

    Support Vector Machine: Training score - 0.8533199565938453 - Test score - 0.8501320557705301
    


```python
performance_df = pd.DataFrame(performance)
```


```python
performance_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>algorithm</th>
      <th>testing_score</th>
      <th>training_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gaussian Naive Bayes</td>
      <td>0.792826</td>
      <td>0.796175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LogisticRegression</td>
      <td>0.795221</td>
      <td>0.798653</td>
    </tr>
    <tr>
      <th>2</th>
      <td>K Neighbors</td>
      <td>0.811989</td>
      <td>0.864765</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Random Forests</td>
      <td>0.948468</td>
      <td>0.996089</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Support Vector Machine</td>
      <td>0.850132</td>
      <td>0.853320</td>
    </tr>
  </tbody>
</table>
</div>



# Implementation of SVM For Spam Mail Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset using chardet
2. Get the dataset info and check for null values
3. Assign x and y values
4. Split it into train and test data
5. Import count vectorizer and transform x_train and x_test as vectors
6. Import SVC and fit it to data
7. Find y_predict values and accuracy
## Program:
```txt
Program to implement the SVM For Spam Mail Detection..
Developed by: Krupa Varsha P
RegisterNumber:  212220220022
```
```python3
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result=chardet.detect(rawdata.read(100000))
result 
```
```python3
import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
```
```python3
data.info()
```
```python3
data.isnull().sum()
```
```python3
x=data["v1"].values
y=data["v2"].values
```
```python3
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
```
```python3
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![Screen Shot 2023-06-02 at 11 42 19 PM](https://github.com/Krupa-Varsha-P/ML_9/assets/100466625/c9503838-d059-46d9-8b4f-2428d844ef4c)
![Screen Shot 2023-06-02 at 11 42 27 PM](https://github.com/Krupa-Varsha-P/ML_9/assets/100466625/52986c05-bf99-45d6-b60d-28bc76d54801)
![Screen Shot 2023-06-02 at 11 42 36 PM](https://github.com/Krupa-Varsha-P/ML_9/assets/100466625/4688301c-f189-4f7b-8c92-6b564cb9e481)
![Screen Shot 2023-06-02 at 11 42 45 PM](https://github.com/Krupa-Varsha-P/ML_9/assets/100466625/3ca5087a-cf15-4cc5-991a-ee36c271375e)
![Screen Shot 2023-06-02 at 11 42 59 PM](https://github.com/Krupa-Varsha-P/ML_9/assets/100466625/9f810c63-966c-489f-bd2c-74257b852b34)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

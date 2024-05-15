# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Load the dataset and check for null data values and duplicate data values in the dataframe.
3. Import label encoder from sklearn.preprocessing to encode the dataset.
4. Apply Logistic Regression on to the model.
5. Predict the y values.
6. Calculate the Accuracy,Confusion and Classsification report.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUBASH E
RegisterNumber:  2122203040209

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
![1a](https://user-images.githubusercontent.com/94747031/194789989-b635e618-4479-46ee-8e38-7b5b489980d0.png)
![2b](https://user-images.githubusercontent.com/94747031/194789999-463f6b8a-7cd1-4dd6-86ac-b0f7e8272f13.png)
![3c](https://user-images.githubusercontent.com/94747031/194790021-b3628ce8-ae59-4f6e-84c8-dd588d84901a.png)
![4d](https://user-images.githubusercontent.com/94747031/194790037-f8b63a18-653e-46ad-9eba-262b6cfb21da.png)
![5e](https://user-images.githubusercontent.com/94747031/194790044-a48cf25e-746e-4bfc-88f8-79cc73724045.png)
![6f](https://user-images.githubusercontent.com/94747031/194790065-4eb7db36-7c2f-4c73-8b1c-52056ce0be91.png)
![7g](https://user-images.githubusercontent.com/94747031/194790119-a44c5803-6b68-4b45-a4b3-c7b2f60b3881.png)

![8h](https://user-images.githubusercontent.com/94747031/194790129-7d166795-254b-4ba2-9eb3-a9589755cbdf.png)
![9i](https://user-images.githubusercontent.com/94747031/194790140-8994904b-a116-4f55-a152-7a64c7b12176.png)
![10j](https://user-images.githubusercontent.com/94747031/194790152-a23e2bc9-3c50-4567-ad71-6c13ceafaa05.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

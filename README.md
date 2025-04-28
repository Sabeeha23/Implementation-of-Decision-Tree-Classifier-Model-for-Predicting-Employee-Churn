# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:  Sabeeha Shaik
RegisterNumber:  212223230176
*/
```
```
import pandas as pd
df = pd.read_csv('Employee.csv')
```
```
df.head()
```
![image](https://github.com/user-attachments/assets/7f8fe687-8efb-4b44-9a6c-3c7334791597)
```
df.isnull().sum()
```

![image](https://github.com/user-attachments/assets/30f3e489-3d8e-496b-a923-5bf091221a6c)
```
df["left"].value_counts()
```

![image](https://github.com/user-attachments/assets/243fb96b-672b-4d0d-b212-74d094704af9)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
```
```
df["salary"]=le.fit_transform(df["salary"])
df.head()
```

![image](https://github.com/user-attachments/assets/b0756a5b-9d4c-4e83-889b-bb0ada081601)
```
x = df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```

![image](https://github.com/user-attachments/assets/e3afa16e-5439-4f49-a129-ac9eca17fae9)
```
y = df["left"]
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
```
```
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/59513da9-cebe-4fab-82dd-b201a2bd2cf4)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print("Sabeeha Shaik")
print(212223230176)
```

![image](https://github.com/user-attachments/assets/4870e1f4-48ed-4433-a4e2-02cd28b895d7)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

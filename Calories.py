import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
df_exercise=pd.read_csv('exercise.csv')
df_calories=pd.read_csv('calories.csv')
df=pd.concat([df_exercise,df_calories['Calories']],axis=1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
X=df.drop(columns=['User_ID','Calories'])
y=df['Calories']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
model=XGBRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(metrics.r2_score(y_pred,y_test))
g=input("Enter Your sex: ")
gender=-1
if g.lower()=='male':
    gender=1
elif g.lower()=='female':
    gender=0
else:
    gender=-1

age=int(input("Enter your age: "))
height=float(input("Enter your Height: "))
weight=float(input("Enter your Weight: "))
duration=float(input("Enter your Duration of Exercise in Minutes: "))
heart=float(input("Enter your Heart Rate after Exercise: "))
Body_temp=float(input("Enter your Body Temperature: "))

X_list=[gender,age,height,weight,duration,heart,Body_temp]
X_df=pd.DataFrame([X_list])
X_df.columns=['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
y_pred_data=model.predict(X_df)
print("Congratulations!!!\nYou Burnt %.3f CALORIES Today"%y_pred_data)





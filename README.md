# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn
7. .Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: kavisree s
RegisterNumber: 21222247001
*/
```
import pandas as p   
data=pd.read_csv("/content/Placement_Data.csv")   
data.head()  
data1=data.copy()  
data1.head()  
data1=data1.drop(['sl_no','salary'],axis=1)  
from sklearn.preprocessing import LabelEncoder  
le=LabelEncoder()   
data1["gender"]=le.fit_transform(data1["gender"])  
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])  
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])  
data1["degree_t"]=le.fit_transform(data1["degree_t"])   
data1["workex"]=le.fit_transform(data1["workex"])   
data1["specialisation"]=le.fit_transform(data1["specialisation"])   
data1["status"]=le.fit_transform(data1["status"])  
data1   
x=data1.iloc[:,:-1]   
x  
y=data1["status"]  
y  
from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)  
from sklearn.linear_model import LogisticRegression   
model=LogisticRegression(solver='liblinear')   
model.fit(x_train,y_train)   
y_pred=model.predict(x_test)   
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report   
accuracy=accuracy_score(y_test,y_pred)  
confusion=confusion_matrix(y_test,y_pred)   
cr=classification_report(y_test,y_pred)   
print("Accuracy Score:",accuracy)  
print("\nConfusion MAtrix:\n,confusion")  
print("\nclassification Report:\n",cr)   
from sklearn import metrics   
cm_display=metrics.ConfusionMatrixDisplay(confusion,display_labels=[True,False])   
cm_display.plot()   

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)

<img width="831" alt="image" src="https://github.com/kavisree86/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145759687/9a29280e-dee6-494f-9a15-51e8a26eba68">

<img width="803" alt="image" src="https://github.com/kavisree86/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145759687/7a368353-8c58-471a-b124-8129043f0765">

<img width="761" alt="image" src="https://github.com/kavisree86/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145759687/1e9a60e7-a21e-49c5-a0ca-ea277474034b">

<img width="747" alt="image" src="https://github.com/kavisree86/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145759687/60b02dbb-d755-49d1-b70b-bad68b77ba25">
<img width="401" alt="image" src="https://github.com/kavisree86/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145759687/648e6fa4-892d-4fbd-ab70-05127f90a771">

<img width="431" alt="image" src="https://github.com/kavisree86/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145759687/6fbcf88c-9caa-41d6-b0cb-ae4e439daa88">








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

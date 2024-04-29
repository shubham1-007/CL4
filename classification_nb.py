import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score,recall_score,classification_report, f1_score

data=pd.read_csv("Iris.csv")
encoder=preprocessing.LabelEncoder()
data['Species']=encoder.fit_transform(data['Species'])


x=data.iloc[:,[1,4]].values
y=data.iloc[:,5].values

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20, random_state=1)

classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)

acc=accuracy_score(y_test,y_pred)
print(acc)

pr=precision_score(y_test,y_pred,average=None)
pr

recall=recall_score(y_test,y_pred,average=None)
recall

cl_report=classification_report(y_test,y_pred)
cl_report

f1=f1_score(y_test,y_pred,average=None)
print(f1)
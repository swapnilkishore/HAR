from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np


file='inputtrain.xlsx'
x1=pd.ExcelFile(file)
df1=x1.parse('Sheet1')
df1.apply(pd.to_numeric, errors='ignore')
file='targettrain.xlsx'
x2=pd.ExcelFile(file)
df2=x2.parse('Sheet1')
df2.apply(pd.to_numeric, errors='ignore')
file='inputtest.xlsx'
x3=pd.ExcelFile(file)
df3=x3.parse('Sheet1')
df3.apply(pd.to_numeric, errors='ignore')
file='targettest.xlsx'
x4=pd.ExcelFile(file)
df4=x4.parse('Sheet1')
df4.apply(pd.to_numeric, errors='ignore')
df1=np.array(df1)
df2=np.array(df2)
df3=np.array(df3)
df4=np.array(df4)

clf = tree.DecisionTreeClassifier()

model=clf.fit(df1, df2)
predictions=clf.predict(df3)
score = accuracy_score(df4, predictions)
print(score)

import numpy as np
import pandas as pd
from sklearn import svm
file='inputtrain.xlsx'
x1=pd.ExcelFile(file)
df1=x1.parse('Sheet1')
df1.apply(pd.to_numeric, errors='ignore')
df1max, df1min = df1.max(), df1.min()
df=(df1-df1min)/(df1max-df1min)
file='targettrain.xlsx'
x2=pd.ExcelFile(file)
df2=x2.parse('Sheet1')
df2.apply(pd.to_numeric, errors='ignore')
file='inputtest.xlsx'
x3=pd.ExcelFile(file)
df3=x3.parse('Sheet1')
df3.apply(pd.to_numeric, errors='ignore')
df3max, df3min = df3.max(), df3.min()
df_test=(df3-df3min)/(df3max-df3min)
df=np.array(df)
df2=np.array(df2)
df_test=np.array(df_test)
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.svc(kernel='linear', c=1, gamma=1) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(df, df2)
model.score(df, df2)
#Predict Output
predicted= model.predict(df_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils import np_utils


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
#file='targettest.xlsx'
#x4=pd.ExcelFile(file)
#df4=x4.parse('Sheet1')
#df4.apply(pd.to_numeric, errors='ignore')
df1=np.array(df1)
df2=np.array(df2)
df3=np.array(df3)
#df4=np.array(df4)


df2=np_utils.to_categorical(df2, num_classes=7)


# create model
model = Sequential()
model.add(Dense(20, input_dim=561, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='sigmoid'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(df1, df2)
score = model.evaluate(df3, df4)
print(score)

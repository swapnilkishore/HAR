import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
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
df=np.array(df)
df2=np.array(df2)
pca = PCA(n_components=561)
pca.fit(df)
X1=pca.fit_transform(df)
#df1=df1.transpose()
#df2=df2.transpose()

#Input array
#df1=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
#df2=np.array([[1],[1],[0]])

#Tansig Function
def tansig (x):
    return 2/(1 + np.exp(-2*x))-1

#Derivative of Sigmoid Function
def derivatives_tansig(x):
    return 4 * np.exp(2*x)/pow((np.exp(2*x) + 1),2)

#Variable initialization
epoch=1000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons =df1.shape[1] #number of features in data set
hiddenlayer_neurons = 7351 #number of hidden layers neurons
output_neurons = 6 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

    #Forward Propogation
    hidden_layer_input1=np.dot(X1,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = tansig(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = tansig(output_layer_input)


    #Backpropagation
    E = df2-output
    slope_output_layer = derivatives_tansig(output)
    slope_hidden_layer = derivatives_tansig(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X1.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print (output)

 

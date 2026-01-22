import numpy as np
import math as mt
from src.logger import logger
from src.components.data_preperation import X_train,y_train

#all the functions
relu = lambda x: np.maximum(0, x)
def softmax(mat):
    exp_scores = np.exp(mat)
    sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp
    return probabilities
#loading the data
Xtrainm=X_train.to_numpy()
ytrainm=y_train.to_numpy()
y_hot = np.eye(6)[ytrainm]

###
from sklearn.preprocessing import StandardScaler

# 1. Initialize
scaler = StandardScaler()

# 2. Scale the Training Data (Learn mean/std from here)
Xtrainm = scaler.fit_transform(Xtrainm)
 ####

#the two basic matrix operations 
#1)need to multiply x matrix with w matrix
#2)need to add b matrix to resultant

#creating neural network
#we will have two layers 
#first we will do normal matrix operations then use relu function on the values
#then we will take those values then matrix operations again and then we use sigmoid function on them 
# so our data has the size 959*11 so 11 features 

# FIRST LAYER
#the layer will have 8 nodes
# so w matrix will be 11*8 
features = 11
neurons_1 = 64
W_1= np.random.randn(features,neurons_1) * 0.01
b_1 = np.zeros((1, neurons_1))

# SECOND LAYER
#the layer will have 6 nodes cuz it is the output layer 
# so w will be 8*6
output=6
W_2= np.random.randn(neurons_1,output) * 0.01
b_2=np.zeros((1, output))


#prediction function
def predict(mat):
    #first layer
    saved_a_1=np.matmul(mat,W_1)
    saved_a_1=saved_a_1+b_1
    saved_a_1=relu(saved_a_1)
    #secondlayer
    a_2=np.matmul(saved_a_1,W_2)
    a_2=a_2+b_2
    ans=softmax(a_2)
    return ans,saved_a_1

 
def lossfxn(ans):
    correct_logprobs = -np.log(ans[range(ytrainm.shape[0]), ytrainm])
    loss = np.sum(correct_logprobs) / ytrainm.shape[0]
    return loss

#code for updating the weight of the first node of output layer
#agg=0
#for i in range(ytrainm.shape[0]):
    #if ytrainm[i]==0:
        #agg-=saved_a_1[i]
    #agg+=ans[i][0]*saved_a_1[i]

learning_rate=0.1
sav_W_2=np.zeros((neurons_1,output))

def training(W_1, b_1, W_2, b_2, X, Y, learning_rate):
    sigmoidmat,sa1=predict(X)
    one_hot_saved_a_1=(sa1 != 0).astype(int)
    loss=lossfxn(sigmoidmat) 
    output_error_matrix=np.zeros((Xtrainm.shape[0],output))
    m = Xtrainm.shape[0]
    output_error_matrix = (Y - sigmoidmat) / m
    sav_W_2=W_2.copy()
    W_2+= learning_rate*np.matmul(np.transpose(sa1),output_error_matrix)
    b_2+=learning_rate*np.sum(output_error_matrix, axis=0, keepdims=True)
    input_error_matrix=np.matmul(output_error_matrix,np.transpose(sav_W_2))*one_hot_saved_a_1
    W_1+=learning_rate*np.matmul(np.transpose(Xtrainm),input_error_matrix)
    b_1+=learning_rate*np.sum(input_error_matrix, axis=0, keepdims=True)
    return loss,W_1, b_1, W_2, b_2

logger.info('running training')
for i in range(5000):
    loss,W_1, b_1, W_2, b_2 = training(W_1, b_1, W_2, b_2, Xtrainm, y_hot, learning_rate)
    if i%100==0:
        logger.info(loss)
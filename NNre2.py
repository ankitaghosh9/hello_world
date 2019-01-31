import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#loading data
data=sio.loadmat('ex4data1.mat')
Y=data['y']
Y[Y==10]=0
X = data['X']
X[X==10]=0
X=(X.T)/255
#print("y size ="+ str(Y.shape))
print("X size ="+str(X.shape))

#one-hot of Y matrix
Y_expand=np.zeros((np.size(Y,0),10))
m=np.size(Y,0)
for i in range(0,m):
 j=Y[i]
 Y_expand[i][j]=1
Y_expand=Y_expand.T
print("Y shape ="+str(Y_expand.shape))

#initializing hyperparameters
n_x = 400
n_h = 30
n_y = 10
m= 5000
learning_rate=0.01
costs = []                        

#parameter initialization
W1 = np.random.randn(n_h,n_x)*0.01
b1 = np.zeros((n_h,1))
W2 = np.random.randn(n_y,n_h)*0.01
b2 = np.zeros((n_y,1))


for i in range(2000):
 #forward propagation
 Z1 = np.dot(W1,X)+b1
 A1 = np.maximum(0,Z1)
 Z2 = np.dot(W2,A1)+b2
 A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

 #cost function
 J= -(1/m)*(np.sum(np.multiply(Y_expand, np.log(A2))))

 #relu derivative
 gZ1=np.array(A1,copy=True)
 gZ1[gZ1>0]=1

 #backprop
 dZ2 = A2-Y_expand
 dW2 = (1/m) * np.dot(dZ2, A1.T)
 db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

 dA1 = np.dot(W2.T, dZ2)
 dZ1 = dA1*gZ1
 dW1 = (1/m) * np.dot(dZ1, X.T)
 db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

 #updating parameters
 W2 = W2 - (learning_rate * dW2)
 b2 = b2 - (learning_rate * db2)
 W1 = W1 - (learning_rate * dW1)
 b1 = b1 - (learning_rate * db1)

 if (i % 100 == 0):
  print("Iteration: "+str(i)+" Cost: "+str(J))
  costs.append(J)

print("Final cost:"+str(J))

# plot the cost
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

"""#testing
Z1 = np.matmul(W1, x_test) + b1
A1 = sigmoid(Z1)
Z2 = np.matmul(W2, A1) + b2
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

predictions = np.argmax(A2, axis=0)
labels = np.argmax(y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))"""









import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np 
import matplotlib.pyplot as plt

x_input = np.arange(0.,50.,.01)
y_input = map(lambda x : 1.337 * np.sin(2.*np.pi*x*1.),x_input)

print 'x_input shape=' + str(len(x_input)) + ', y_input shape=' + str(len(y_input))

lenOfTime = 30 # number of periods in the past to base the predictions on
stepsInFuture = 20 # number of periods in the future to predict

for i in range(3000):
    _mx = np.hstack((y_input[i:(i+lenOfTime)], np.repeat(y_input[i+lenOfTime-1],stepsInFuture)))
    _target = y_input[(i+lenOfTime):(i+lenOfTime+stepsInFuture)]
    
    if i == 0:
        mx = _mx
        target = _target
        #print 'mx 0 = ' + str(mx) + ', target 0 = ' + str(target)
    else:
        mx = np.vstack((mx,_mx))
        target = np.vstack((target,_target))  

print 'mx shape=' + str(mx.shape) + ', target shape=' + str(target.shape)        
 
X = mx.reshape(3000,lenOfTime+stepsInFuture,1)
y = target.reshape(3000,stepsInFuture,1)        

print 'X shape=' + str(X.shape) + ', y shape=' + str(y.shape) 

X_train = X[0:2000,:]
y_train = y[0:2000,:]
X_test = X[2000:,:]
y_test = y[2000:,:]

print 'X_train shape=' + str(X_train.shape) + ', y_train shape=' + str(y_train.shape) 
print 'X_test shape=' + str(X_test.shape) + ', y_test shape=' + str(y_test.shape) 
#print 'X_train[0:3,:,0] = ' + str(X_train[0:3,:,0]) + ', y_train[0:3,:,0] = ' + str(y_train[0:3,:,0])

hidden_neurons = 100
def time_slice(output):
    return output[:,-20:,:] #todo: how to use stepsInFuture here?
    
model = Sequential()  
model.add(LSTM(output_dim=hidden_neurons, input_dim = 1, return_sequences=True, activation='tanh'))
model.add(LSTM(output_dim=hidden_neurons, input_dim = hidden_neurons, return_sequences=True, activation='tanh'))
model.add(LSTM(output_dim=hidden_neurons, input_dim = hidden_neurons, return_sequences=True, activation='tanh'))
model.add(Lambda(time_slice, output_shape=(stepsInFuture, hidden_neurons)))
model.add(TimeDistributedDense(output_dim=1, activation = 'linear', input_dim=hidden_neurons))
start = time.time()
model.compile(loss="mse", optimizer="rmsprop") 
print 'model compiled in ' + str(time.time() - start) + ' seconds'
# model.fit(X_train, y_train, batch_size=128, nb_epoch=30,validation_data=(X_test, y_test), show_accuracy=True)

# plt.plot(model.predict(X_test)[0,:,0], label='predicted')
# plt.plot(y_test[0,:,0], label = 'actual')
# plt.legend()
# plt.show()

# print "MAE: {0:.6f}".format(np.mean(abs(y_test - model.predict(X_test))))
# print "MSE: {0:.6f}".format(np.mean((y_test - model.predict(X_test)) ** 2.))

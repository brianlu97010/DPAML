import os
import time
import tensorflow as tf
import scipy.io as sio 
from sklearn.model_selection import train_test_split
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

def model(input_shape,class_num):
    
    #activation = 'relu'
    
    X_input = Input(input_shape)
    X = Conv2D(32, (1, 1), strides = (1, 1), name = 'conv01')(X_input)
    X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv02')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    X = Dropout(0.5)(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)
    
    X = Conv2D(64, (1, 1), strides = (1, 1), name = 'conv11')(X)
    X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv12')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    X = Conv2D(128, (1, 1), strides = (1, 1), name = 'conv21')(X)
    X = Conv2D(256, (3, 3), strides = (1, 1), name = 'conv22')(X)
    X = Conv2D(512, (3, 3), strides = (1, 1), name = 'conv23')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2, 2), name='max_pool3')(X)
    
    X = Dropout(0.3)(X)
    
    X = Flatten()(X)
    X = Dense(512, activation='sigmoid', name='fc0')(X)
    X = Dense(3, activation='softmax', name='fc1')(X)
    
    model = Model(inputs = X_input, outputs = X, name='DrawModel')
    return model

def one_hot_matrix(labels, C):
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'.
    depth = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, depth, axis=0)
    
    # Create the session
    sess = tf.Session()
    
    # Run the session
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session.
    sess.close() 
    
    return one_hot

#--------------------------------------------------------

file_path = os.getcwd() + '\\train'
label_name = os.listdir(file_path)
class_num = len(label_name)

matfn='data.mat' 
data=sio.loadmat(matfn) 

X = data['X_data'] 
Y = data['Y_data'] 

data_num = X.shape[0] 

X = X[:,:,:,np.newaxis] 
Y = one_hot_matrix(Y, class_num)
Y = Y.T
Y = Y.reshape(data_num,class_num)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


DrawModel = model((64,64,1), class_num)
DrawModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
DrawModel.fit(x=X_train, y=Y_train, validation_split=0.2, epochs=50, batch_size=32)

complete_time=time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) 
DrawModel.save('DrawModel_'+str(complete_time)+'.h5')

preds = DrawModel.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

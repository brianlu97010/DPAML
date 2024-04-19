import scipy.io as sio 
import numpy as np
import cv2
import os

train_file_path = os.getcwd() + '\\train'
all_file_name = os.listdir(train_file_path)

Y_data = []

for i in range(len(all_file_name)):
    pic_file_path = train_file_path + '\\' + all_file_name[i] 
    all_pic_name = os.listdir(pic_file_path)
    for j in range(len(all_pic_name)):
        pic_path = pic_file_path + '\\' + all_pic_name[j]
        img = cv2.imread(pic_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        img = cv2.bitwise_not(thresh)
        if i == 0 and j == 0:
            X_data = img
            X_data = np.expand_dims(X_data, axis = 0)
        else:
            X_data = np.vstack((X_data,img[np.newaxis,:,:]))
        Y_data.append(i)
        print(str(i) + '-' + str(j))  #查看進度
Y_data = np.array(Y_data)

save_train = 'data.mat'
save_array_x = X_data
save_array_y = Y_data
sio.savemat(save_train, {'X_data': save_array_x, 'Y_data': save_array_y})    
          
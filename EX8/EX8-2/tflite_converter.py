# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:10:12 2021

@author: user
"""

import tensorflow as tf

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model_file('DrawModel_2021_11_30 03_20_40.h5')
tflite_model = converter.convert()

# Save the model.
with open('DrawModel_64x64.tflite', 'wb') as f:
  f.write(tflite_model)
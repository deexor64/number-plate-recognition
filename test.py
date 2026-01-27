import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


x = tf.constant([[1, 3], [3, 5]])


print(tf.config.list_physical_devices('TPU'))

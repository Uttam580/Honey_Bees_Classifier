
# to test gpu 
import tensorflow as tf 
from tensorflow.python.client import device_lib

print(tf.test.is_built_with_cuda())
print(device_lib.list_local_devices())
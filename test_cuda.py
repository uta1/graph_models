import tensorflow as tf
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('physical devices:', str(tf.config.experimental.list_physical_devices()))

import tensorflow as tf
device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
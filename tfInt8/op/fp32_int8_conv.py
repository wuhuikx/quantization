#python3 compatibility
from __future__ import print_function

#disable warnings from h5py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.quantize.python import quantize

tf.set_random_seed(204)
np.random.seed(204)
n = 4
ic = 3
iw = 32
ih = 32
oc = 16
kernel_ = 3
stride_ = 1
pad_ = "valid"

dummy_data = np.random.normal(size=(n, iw, ih, ic))
inputs = tf.placeholder(tf.float32, [n, iw, ih, ic])
'''
conv = tf.layers.conv2d(inputs,
                        filters = oc,
                        kernel_size = kernel_,
                        strides = stride_,
                        padding = pad_,
                        use_bias = False)


init = tf.global_variables_initializer()

# FP32 conv forward.
with tf.Session() as sess:
    sess.run(init)
    fp32_data = sess.run(inputs, feed_dict={inputs: dummy_data})
    fp32_start = time.time()
    fp32_conv = sess.run(conv,   feed_dict={inputs: dummy_data})
    fp32_end = time.time()
'''

#----------------------------- fp32 conv forward---------------------
init = tf.global_variables_initializer()
saver = tf.train.Saver()

conv = tf.contrib.layers.conv2d(inputs,
                        num_outputs = oc,
                        kernel_size = kernel_,
                        stride = stride_,
                        padding = pad_)

with tf.Session() as sess:
     sess.run(init)
     #fp32_data = sess.run(inputs, feed_dict = {inputs: dummy_data})
     start = time.time()
     fp32_conv = sess.run(conv, feed_dict = {inputs: dummy_data})
     end = time.time()
     saver.save(sess, "checkpoint/conv")

     print("elapsed time: %.8f sec" % (end -start))


#----------------------------- Int8 conv forward---------------------
graph = tf.get_default_graph()
quantize.Quantize(graph, is_training=False, weight_bits=8, activation_bits=8)
init = tf.global_variables_initializer()
with tf.Session() as sess:
     sess.run(init)
     convq = graph.get_tensor_by_name("Conv/act_quant/FakeQuantWithMinMaxVars:0")

     start_int8 = time.time()
     int8_conv = sess.run(convq, feed_dict={inputs: dummy_data})
     end_int8 = time.time()

     print("elapsed time: %.8f sec" % (end -start))



"""
https://blog.csdn.net/weixin_41560402/article/details/106930463

在 pytorch 中，梯度只要不清零默认是累加的，于是很容易实现上述问题。
但在Tensorflow中，却不那么容易。话不多说，直接上程序。

"""

import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

x_data = np.array(range(1, 20))
num_dataset = len(x_data)
batch_size = 4
minibatch_size = 2
with tf.Graph().as_default():
    x = tf.placeholder(dtype='float32', shape=None)
    w = tf.Variable(initial_value=4., dtype='float32')
    loss = w * w * x

    # Optimizer definition - nothing different from any classical example
    opt = tf.train.GradientDescentOptimizer(0.1)

    # Retrieve all trainable variables you defined in your graph
    tvs = tf.trainable_variables()

    # Creation of a list of variables with the same shape as the trainable ones
    # initialized with zeros
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

    # Calls the compute_gradients function of the optimizer to obtain the list of gradients
    gvs = opt.compute_gradients(loss, tvs)

    # Adds to each element from the list you initialized earlier with zeros its gradient
    # (works because accum_vars and gvs are in the same order)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    # Define the training step (part with variable value update)
    train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for batch_count in range(batch_size):
            # 在run每个batch, 需先将前一个batch所得的累积梯度清零
            sess.run(zero_ops)

            batch_data = x_data[batch_count*batch_size: (batch_count+1)*batch_size]
            # Accumulate the gradients 'minibatch_size' times in accum_vars using accum_ops
            for minibatch_count in range(minibatch_size):
                minibatch_data = batch_data[minibatch_count*minibatch_size: (minibatch_count+1)*minibatch_size]
                accum_array = sess.run(accum_ops, feed_dict={x: minibatch_data})
                print("[%d][%d]" % (batch_count, minibatch_count), accum_array)
                print(sess.run(tvs))
            # Run the train_step ops to update the weights based on your accumulated gradients
            sess.run(train_step)

tf.estimator.Estimator()
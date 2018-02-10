import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.contrib.layers import fully_connected
from datetime import datetime

def DeepNN(X_train, y_train, X_test, y_test):
    
    #Construction Phase
    now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    
    ##network parameters
    n_inputs = X_train.shape[1] #dimension of dataframe columns
    n_hidden1 = 10
    n_hidden2 = 10
    n_outputs = 2 #dimension of possible outcomes (notcert=0,cert=1)

    ##placeholders
    X = tf.placeholder(X_train.dtype, shape=(None,n_inputs))
    y = tf.placeholder(tf.int64)

    ##Neuron Layers
    with tf.name_scope("dnn"):
        hidden1 = fully_connected(X, n_hidden1)
        hidden2 = fully_connected(hidden1, n_hidden2)
        logits = fully_connected(hidden2, n_outputs, activation_fn=None)

    with tf.name_scope("loss"):
        xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits=logits)
        loss = tf.reduce_mean(xentropy)

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(tf.cast(logits, dtype=tf.float32), tf.cast(y, dtype=tf.int32), 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    loss_summary = tf.summary.scalar('XentropyLoss', loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        
    #Execution Phase
    def batch_getter(batch_size, X, y):
        rand_ind = np.random.randint(0,X.shape[0],batch_size)
        x_batch = X[rand_ind,:]
        y_batch = y[rand_ind]
        return x_batch, y_batch

    n_epochs = 100
    batch_size = 50

    init = tf.global_variables_initializer()
    #train Neural Network
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            X_bat, y_bat = batch_getter(batch_size, X_train, y_train)
            feed = {X:X_bat, y: y_bat}
            sess.run(training_op, feed_dict=feed)
            summary_str = loss_summary.eval(feed_dict=feed)
            file_writer.add_summary(summary_str, epoch)
        acc_train = accuracy.eval(feed_dict=feed)
        acc_test = accuracy.eval(feed_dict = {X:X_test, y:y_test})
    
        
    return acc_train, acc_test 





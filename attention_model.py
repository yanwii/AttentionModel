# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-08-14 17:33:16
'''

import tensorflow as tf
import tensorflow.contrib.layers as layers

def concatenation_based_attention(
    hidden_states, 
    dt, 
    batch_size, 
    num_units,
    initializer=layers.variance_scaling_initializer(), 
    activation_fn=tf.tanh):
    """
        Concatenation-based Attention
        
        args:
            `encoder_hiddens`: the hidden states of encoder which shape is [batch_size, time_steps, num_units]
            `dt`: the hidden state of current decoder 
            `num_units`: num of the LSTM cell's hidden size
    """
    with tf.variable_scope("attenstion") as scope:
        attention_context_vector = tf.get_variable(
            name="attention_context_vector",
            shape=[1, num_units, 1],
            initializer=initializer,
            dtype=tf.float32
        )
        W_1 = tf.get_variable(
            name="W_1",
            shape=[1, num_units, num_units],
            initializer=initializer,
            dtype=tf.float32
        )
        b_1 = tf.get_variable(
            "b_1", 
            shape=[num_units], 
            dtype=tf.float32,
            initializer=initializer
        )

        W_2 = tf.get_variable(
            name="W_2",
            shape=[1, num_units, num_units],
            initializer=initializer,
            dtype=tf.float32
        )

        b_2 = tf.get_variable(
            "b_2", 
            shape=[num_units], 
            dtype=tf.float32,
            initializer=initializer
        )
        W_1 = tf.tile(W_1, [batch_size, 1, 1])
        W_2 = tf.tile(W_2, [batch_size, 1, 1])

        attention_context_vector = tf.tile(attention_context_vector, [batch_size, 1, 1])
        model_inputs = tf.nn.xw_plus_b(hidden_states, W_1, b_1) + tf.nn.xw_plus_b(dt, W_2, b_2)

        input_projection = tf.nn.tanh(model_inputs)
        
        u = tf.matmul(input_projection, attention_context_vector)
        u = tf.reshape(u, shape=[batch_size, 1, -1])
        alpha = tf.nn.softmax(u)
        d = tf.matmul(alpha, hidden_states)
        return d, alpha

if __name__ == "__main__":
    with tf.Session() as sess:
        hidden, alpha = concatenation_based_attention(
            [
                [[0.1, 0.03, 0.003], [4.0, 5.0, 6.0]]
            ],
            [
                [[7.0, 8.0, 9.0]]
            ],
            1,
            3
        )
        sess.run(tf.global_variables_initializer())
        print(hidden.eval())
        print(alpha.eval())
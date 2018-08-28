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
    e_num_units,
    d_num_units,
    attention_size,
    initializer=layers.xavier_initializer(), 
    activation_fn=tf.tanh):
    """
        Concatenation-based Attention
        
        args:
            `hidden_states`: the hidden states of encoder which shape is [batch_size, time_steps, num_units]
            `dt`: the hidden state of current decoder 
            `e_num_units`: num of the encoder hidden size
            `d_num_units`: num of the decoder hidden size
            `attention_size`: size of attention context vector
    """
    with tf.variable_scope("attenstion_concatenation") as scope:
        batch_size = tf.shape(hidden_states)[0]
        attention_context_vector = tf.get_variable(
            name="attention_context_vector",
            shape=[1, attention_size, 1],
            initializer=initializer,
            dtype=tf.float32
        )
        W_1 = tf.get_variable(
            name="W_1",
            shape=[1, e_num_units, attention_size],
            initializer=initializer,
            dtype=tf.float32
        )
        b_1 = tf.get_variable(
            "b_1", 
            shape=[attention_size], 
            dtype=tf.float32,
            initializer=initializer
        )

        W_2 = tf.get_variable(
            name="W_2",
            shape=[1, d_num_units, attention_size],
            initializer=initializer,
            dtype=tf.float32
        )

        b_2 = tf.get_variable(
            "b_2", 
            shape=[attention_size], 
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

def location_based_attention(
    hidden_states,
    num_units,
    initializer=layers.xavier_initializer(), 
    activation_fn=tf.tanh
    ):
    '''
        Location-based Attention

        args:
            `hiddent_state`: the hidden states of encoder which shape is [batch_size, time_steps, num_units]
            `num_units`: num of the LSTM cell's hidden size
    '''

    with tf.variable_scope("attenstion_location") as scope:
        batch_size = tf.shape(hidden_states)[0]
        W_1 = tf.get_variable(
            name="W_1",
            shape=[num_units, 1],
            initializer=initializer,
            dtype=tf.float32
        )
        b_1 = tf.get_variable(
            "b_1", 
            shape=[1], 
            dtype=tf.float32,
            initializer=initializer
        )
        hidden = tf.reshape(hidden_states, shape=[-1, num_units])
        
        model_inputs = tf.nn.xw_plus_b(hidden, W_1, b_1) # [BT x D] * [D x 1] => [BT x 1]
        input_projection = tf.nn.tanh(model_inputs)  # [BT x 1]
 
        u = tf.reshape(input_projection, shape=[batch_size, 1, -1])  # [BT x 1] => [B x 1 x T]
        alpha = tf.nn.softmax(u)  # [B x 1 x T]
        d = tf.matmul(alpha, hidden_states) # [B x 1 x T] * [B x T x D] => [B x 1 x D]
        return d, alpha

if __name__ == "__main__":
    with tf.Session() as sess:
        # hidden, alpha= concatenation_based_attention(
        #     [
        #         [[0.1, 0.03, 0.003], [4.0, 5.0, 6.0]],
        #         [[0.1, 0.03, 0.003], [1.0, 3.0, 1.0]]
        #     ],
        #     [
        #         [[7.0, 8.0, 9.0, 10.0]],
        #         [[7.0, 8.0, 9.0, 10.0]]
        #     ],
        #     3,
        #     4,
        #     100
        # )

        hidden, alpha= location_based_attention(
            [
                [[0.1, 0.03, 0.003], [4.0, 5.0, 6.0]],
                [[0.1, 0.03, 0.003], [1.0, 3.0, 1.0]]
            ],
            3
        )
        sess.run(tf.global_variables_initializer())
        print(hidden.eval())
        print(alpha.eval())
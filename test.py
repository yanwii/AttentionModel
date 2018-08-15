

a = [
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ],
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ],
]

b = [
    [
        [1, 2, 3],
        [0, 0, 0]
    ],
    [
        [1, 2, 3],
        [1, 1, 1]
    ],
]

import tensorflow as tf

a = tf.constant(a, dtype=tf.float32)
b = tf.constant(b, dtype=tf.float32)

sess = tf.Session()

print(sess.run(tf.matmul(a, b)))

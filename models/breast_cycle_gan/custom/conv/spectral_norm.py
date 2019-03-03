# Copyright 2019 Lukas Jendele and Ondrej Skopek.
# Adapted from The TensorFlow Authors, under the ASL 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is copied from:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
import tensorflow as tf


def apply_spectral_norm(x, is_training=False):
    """Normalizes x using the spectral norm.
  The implementation follows Algorithm 1 of
  https://arxiv.org/abs/1802.05957. If x is not a 2-D Tensor, then it is
  reshaped such that the number of channels (last-dimension) is the same.
  Args:
    x: Tensor with the last dimension equal to the number of filters.
  Returns:
    x: Tensor with the same shape as x normalized by the spectral norm.
    assign_op: Op to be run after every step to update the vector "u".
  """
    weights_shape = shape_list(x)
    other, num_filters = tf.reduce_prod(weights_shape[:-1]), weights_shape[-1]

    # Reshape into a 2-D matrix with outer size num_filters.
    weights_2d = tf.reshape(x, (other, num_filters))

    # v = Wu / ||W u||
    with tf.variable_scope("u", reuse=tf.AUTO_REUSE):
        u = tf.get_variable("u", [num_filters, 1], initializer=tf.truncated_normal_initializer(), trainable=False)
    v = tf.nn.l2_normalize(tf.matmul(weights_2d, u))

    # u_new = vW / ||v W||
    u_new = tf.nn.l2_normalize(tf.matmul(tf.transpose(v), weights_2d))

    # s = v*W*u
    spectral_norm = tf.squeeze(tf.matmul(tf.transpose(v), tf.matmul(weights_2d, tf.transpose(u_new))))

    # set u equal to u_new in the next iteration.
    assign_op = tf.assign(u, tf.transpose(u_new))
    if is_training:
        with tf.control_dependencies([assign_op]):
            result = tf.divide(x, spectral_norm)
    else:
        result = tf.divide(x, spectral_norm)
    return result


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

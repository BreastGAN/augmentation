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
# This part is copied from:
# https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/layers/python/layers/layers.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import add_arg_scope
# from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
# from tensorflow.python.eager import context
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import function
from tensorflow.python.framework import ops
# from tensorflow.python.framework import sparse_tensor
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

# My imports
from tensorflow.contrib.layers.python.layers.layers import _build_variable_getter, _add_variable_to_collections
from models.breast_cycle_gan.custom.conv.layers import MyConv2D
import tensorflow as tf
# This part is copied from:
# https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/layers/python/layers/layers.py


@add_arg_scope
def convolution2d(inputs,
                  num_outputs,
                  kernel_size,
                  stride=1,
                  padding='SAME',
                  data_format=None,
                  rate=1,
                  activation_fn=nn.relu,
                  normalizer_fn=None,
                  normalizer_params=None,
                  weights_initializer=initializers.xavier_initializer(),
                  weights_regularizer=None,
                  biases_initializer=init_ops.zeros_initializer(),
                  biases_regularizer=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  use_spectral_norm=False,
                  is_training=False,
                  self_attention=False,
                  scope=None):
    h = convolution(
            inputs,
            num_outputs,
            kernel_size,
            stride,
            padding,
            data_format,
            rate,
            activation_fn,
            normalizer_fn,
            normalizer_params,
            weights_initializer,
            weights_regularizer,
            biases_initializer,
            biases_regularizer,
            reuse,
            variables_collections,
            outputs_collections,
            trainable,
            use_spectral_norm,
            is_training,
            scope,
            conv_dims=2)
    if not self_attention:
        return h
    with tf.variable_scope("self_attention"):
        with tf.variable_scope("f"):
            f = convolution(
                    inputs,
                    num_outputs // 8,
                    kernel_size,
                    stride,
                    padding,
                    data_format,
                    rate,
                    activation_fn,
                    normalizer_fn,
                    normalizer_params,
                    weights_initializer,
                    weights_regularizer,
                    biases_initializer,
                    biases_regularizer,
                    reuse,
                    variables_collections,
                    outputs_collections,
                    trainable,
                    use_spectral_norm,
                    is_training,
                    None,
                    conv_dims=2)
        with tf.variable_scope("g"):
            g = convolution(
                    inputs,
                    num_outputs // 8,
                    kernel_size,
                    stride,
                    padding,
                    data_format,
                    rate,
                    activation_fn,
                    normalizer_fn,
                    normalizer_params,
                    weights_initializer,
                    weights_regularizer,
                    biases_initializer,
                    biases_regularizer,
                    reuse,
                    variables_collections,
                    outputs_collections,
                    trainable,
                    use_spectral_norm,
                    is_training,
                    None,
                    conv_dims=2)

        def hw_flatten(x):
            return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

        return x


@add_arg_scope
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                use_spectral_norm=False,
                is_training=False,
                scope=None,
                conv_dims=None):
    """Adds an N-D convolution followed by an optional batch_norm layer.
  It is required that 1 <= N <= 3.
  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
  provided (such as `batch_norm`), it is then applied. Otherwise, if
  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
  variable would be created and added the activations. Finally, if
  `activation_fn` is not `None`, it is applied to the activations as well.
  Performs atrous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.
  Args:
    inputs: A Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with "NC" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with "NC".
    num_outputs: Integer, the number of output filters.
    kernel_size: A sequence of N positive integers specifying the spatial
      dimensions of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: A sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: One of `"VALID"` or `"SAME"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with "NC"), or the second dimension (if `data_format`
      starts with "NC").  For N=1, the valid values are "NWC" (default) and
      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW".
      For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    rate: A sequence of N positive integers specifying the dilation rate to use
      for atrous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.
    conv_dims: Optional convolution dimensionality, when set it would use the
      corresponding convolution (e.g. 2 for Conv 2D, 3 for Conv 3D, ..). When
      leaved to None it would select the convolution dimensionality based on
      the input rank (i.e. Conv ND, with N = input_rank - 2).
  Returns:
    A tensor representing the output of the operation.
  Raises:
    ValueError: If `data_format` is invalid.
    ValueError: Both 'rate' and `stride` are not uniformly 1.
  """
    if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))

    layer_variable_getter = _build_variable_getter({'bias': 'biases', 'kernel': 'weights'})

    with variable_scope.variable_scope(scope, 'Conv', [inputs], reuse=reuse, custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)
        input_rank = inputs.get_shape().ndims

        if conv_dims is not None and conv_dims + 2 != input_rank:
            raise ValueError('Convolution expects input with rank %d, got %d' % (conv_dims + 2, input_rank))
        if input_rank == 3:
            layer_class = convolutional_layers.Convolution1D
        elif input_rank == 4:
            layer_class = MyConv2D
        elif input_rank == 5:
            layer_class = convolutional_layers.Convolution3D
        else:
            raise ValueError('Convolution not supported for input with rank', input_rank)

        df = ('channels_first' if data_format and data_format.startswith('NC') else 'channels_last')
        layer = layer_class(
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                data_format=df,
                dilation_rate=rate,
                activation=None,
                use_bias=not normalizer_fn and biases_initializer,
                kernel_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                kernel_regularizer=weights_regularizer,
                bias_regularizer=biases_regularizer,
                activity_regularizer=None,
                use_spectral_norm=use_spectral_norm,
                is_training=is_training,
                trainable=trainable,
                name=sc.name,
                dtype=inputs.dtype.base_dtype,
                _scope=sc,
                _reuse=reuse)
        outputs = layer.apply(inputs)

        # Add variables to collections.
        _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
        if layer.use_bias:
            _add_variable_to_collections(layer.bias, variables_collections, 'biases')

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

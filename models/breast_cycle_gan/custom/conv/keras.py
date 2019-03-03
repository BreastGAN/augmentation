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
# https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/keras/layers/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

from models.breast_cycle_gan.custom.conv import spectral_norm


class MyKerasConv(Layer):
    """Abstract nD convolution layer (private, used as implementation base).
  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_spectral_norm=False,
                 is_training=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MyKerasConv, self).__init__(
                trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_spectral_norm = use_spectral_norm
        self.is_training = is_training
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs ' 'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
                name='kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self._convolution_op = nn_ops.Convolution(
                input_shape,
                filter_shape=self.kernel.get_shape(),
                dilation_rate=self.dilation_rate,
                strides=self.strides,
                padding=self.padding.upper(),
                data_format=conv_utils.convert_data_format(self.data_format, self.rank + 2))
        self.built = True

    def call(self, inputs):
        if self.use_spectral_norm:
            print("using spectral norm")
            norm_kernel = spectral_norm.apply_spectral_norm(self.kernel, self.is_training)
        else:
            print("not using spectral norm")
            norm_kernel = self.kernel
        outputs = self._convolution_op(inputs, norm_kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(
                            outputs,
                            [outputs_shape[0], outputs_shape[1], outputs_shape[2] * outputs_shape[3], outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                        space[i],
                        self.kernel_size[i],
                        padding=self.padding,
                        stride=self.strides[i],
                        dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] + new_space)

    def get_config(self):
        config = {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'activation': activations.serialize(self.activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(self.kernel_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'use_spectral_norm': self.use_spectral_norm,
                'is_training': self.is_training,
                'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                'kernel_constraint': constraints.serialize(self.kernel_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MyKerasConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyKerasConv2D(MyKerasConv):
    """2D convolution layer (e.g. spatial convolution over images).
  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          height and width of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the height and width.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
  Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 use_spectral_norm=False,
                 is_training=False,
                 bias_constraint=None,
                 **kwargs):
        super(MyKerasConv2D, self).__init__(
                rank=2,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activations.get(activation),
                use_bias=use_bias,
                use_spectral_norm=use_spectral_norm,
                is_training=is_training,
                kernel_initializer=initializers.get(kernel_initializer),
                bias_initializer=initializers.get(bias_initializer),
                kernel_regularizer=regularizers.get(kernel_regularizer),
                bias_regularizer=regularizers.get(bias_regularizer),
                activity_regularizer=regularizers.get(activity_regularizer),
                kernel_constraint=constraints.get(kernel_constraint),
                bias_constraint=constraints.get(bias_constraint),
                **kwargs)

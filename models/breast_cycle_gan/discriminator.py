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
"""Implementation of the Image-to-Image Translation model.

This network represents a port of the following work:

  Image-to-Image Translation with Conditional Adversarial Networks
  Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
  Arxiv, 2017
  https://phillipi.github.io/pix2pix/

A reference implementation written in Lua can be found at:
https://github.com/phillipi/pix2pix/blob/master/models.lua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.breast_cycle_gan.custom.conv.contrib import convolution2d

layers = tf.contrib.layers


def pix2pix_arg_scope(use_spectral_norm=False, is_training=False):
    """Returns a default argument scope for isola_net.

  Returns:
    An arg scope.
  """
    # These parameters come from the online port, which don't necessarily match
    # those in the paper.
    # TODO(nsilberman): confirm these values with Philip.
    instance_norm_params = {
            'center': True,
            'scale': True,
            'epsilon': 0.00001,
    }

    with tf.contrib.framework.arg_scope([convolution2d, layers.conv2d_transpose],
                                        normalizer_fn=layers.instance_norm,
                                        normalizer_params=instance_norm_params,
                                        weights_initializer=tf.random_normal_initializer(0, 0.02),
                                        use_spectral_norm=use_spectral_norm,
                                        is_training=is_training) as sc:
        return sc


def pix2pix_discriminator(net, num_filters, padding=2, self_attention=False):
    """Creates the Image2Image Translation Discriminator.

  Args:
    net: A `Tensor` of size [batch_size, height, width, channels] representing
      the input.
    num_filters: A list of the filters in the discriminator. The length of the
      list determines the number of layers in the discriminator.
    padding: Amount of reflection padding applied before each convolution.
    is_training: Whether or not the model is training or testing.

  Returns:
    A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of
    'patches' we're attempting to discriminate and a dictionary of model end
    points.
  """
    end_points = {}

    num_layers = len(num_filters)

    def padded(net, scope):
        if padding:
            with tf.variable_scope(scope):
                spatial_pad = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]], dtype=tf.int32)
                return tf.pad(net, spatial_pad, 'REFLECT')
        else:
            return net

    with tf.contrib.framework.arg_scope([convolution2d],
                                        kernel_size=[4, 4],
                                        stride=2,
                                        padding='valid',
                                        activation_fn=tf.nn.leaky_relu):

        # No normalization on the input layer.
        print("input before conv0", net.get_shape())
        net = convolution2d(padded(net, 'conv0'), num_filters[0], normalizer_fn=None, scope='conv0')

        end_points['conv0'] = net

        for i in range(1, num_layers - 1):
            net = convolution2d(padded(net, 'conv%d' % i), num_filters[i], scope='conv%d' % i)
            end_points['conv%d' % i] = net

        if self_attention:
            with tf.variable_scope("self_attention1"):
                print("Shape before self attention1: ", net.get_shape())
                net = convolution2d(
                        net,
                        num_filters[-2],
                        stride=1,
                        kernel_size=[1, 1],
                        activation_fn=None,
                        self_attention=True,
                        padding='VALID')
                print("Shape after self attention1: ", net.get_shape())
        # Stride 1 on the last layer.
        net = convolution2d(
                padded(net, 'conv%d' % (num_layers - 1)), num_filters[-1], stride=1, scope='conv%d' % (num_layers - 1))
        if self_attention:
            with tf.variable_scope("self_attention2"):
                print("Shape before self attention1: ", net.get_shape())
                net = convolution2d(
                        net,
                        num_filters[-1],
                        stride=1,
                        kernel_size=[1, 1],
                        activation_fn=None,
                        self_attention=True,
                        padding='VALID')
                print("Shape after self attention1: ", net.get_shape())
        end_points['conv%d' % (num_layers - 1)] = net

        # 1-dim logits, stride 1, no activation, no normalization.
        logits = convolution2d(
                padded(net, 'conv%d' % num_layers),
                1,
                stride=1,
                activation_fn=None,
                normalizer_fn=None,
                scope='conv%d' % num_layers)
        end_points['logits'] = logits
        end_points['predictions'] = tf.sigmoid(logits)
    return logits, end_points


def discriminator(image_batch,
                  unused_conditioning=None,
                  use_spectral_norm=False,
                  is_training=False,
                  self_attention=False):
    """A thin wrapper around the Pix2Pix discriminator to conform to TFGAN API."""
    # var_scope_name = tf.get_variable_scope().name
    with tf.contrib.framework.arg_scope(
            pix2pix_arg_scope(is_training=is_training, use_spectral_norm=use_spectral_norm)):
        logits_4d, _ = pix2pix_discriminator(
                image_batch, num_filters=[64, 128, 256, 512], self_attention=self_attention)
        logits_4d.shape.assert_has_rank(4)
    # Output of logits is 4D. Reshape to 2D, for TFGAN.
    logits_2d = tf.layers.flatten(logits_4d)

    return logits_2d

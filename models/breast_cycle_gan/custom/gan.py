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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope

from tensorflow.python.ops.losses import losses
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util

# For custom cyclegan summaries.
from tensorflow.contrib.gan.python.eval.python import eval_utils

import tensorflow as tf


# copied from
#    https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/summaries_impl.py
def add_cyclegan_image_summaries(cyclegan_model, include_masks):
    """Adds image summaries for CycleGAN.
  There are two summaries, one for each generator. The first image is the
  generator input, the second is the generator output, and the third is G(F(x)).
  Args:
    cyclegan_model: A CycleGANModel tuple.
  Raises:
    ValueError: If `cyclegan_model` isn't a CycleGANModel.
    ValueError: If generated data, generator inputs, and reconstructions aren't
      images.
    ValueError: If the generator input, generated data, and reconstructions
      aren't all the same size.
  """

    def _add_comparison_summary(gan_model, reconstructions):
        image_list = [
                gan_model.generator_inputs[0, :, :, 0], gan_model.generated_data[0, :, :, 0],
                reconstructions[0, :, :, 0]
        ]
        image_list = [tf.expand_dims(x, axis=-1) for x in image_list]
        image_diff_list = [image_list[1] - image_list[0], image_list[2] - image_list[1], image_list[2] - image_list[0]]

        print('image', image_list[0].get_shape())
        summary_list = list(image_list)
        summary_list.extend(image_diff_list)
        if include_masks:
            image_list_masks = [
                    gan_model.generator_inputs[0, :, :, 1], gan_model.generated_data[0, :, :, 1],
                    reconstructions[0, :, :, 1]
            ]
            image_list_masks = [tf.expand_dims(x, axis=-1) for x in image_list_masks]
            image_mask_diff_list = [
                    image_list_masks[1] - image_list_masks[0], image_list_masks[2] - image_list_masks[1],
                    image_list_masks[2] - image_list_masks[0]
            ]
            print('mask', image_list_masks[0].get_shape())
            summary_list.extend(image_list_masks)
            summary_list.extend(image_mask_diff_list)

        tf.summary.image(
                'image_comparison', eval_utils.image_reshaper(summary_list, num_cols=len(image_list)), max_outputs=1)

        # tf.summary.image(
        #         'image_diff',
        #         eval_utils.image_reshaper(summary_diff_list, num_cols=len(image_diff_list)),
        #         max_outputs=1)

    with tf.name_scope('H2C_image_comparison_summaries'):
        _add_comparison_summary(cyclegan_model.model_x2y, cyclegan_model.reconstructed_x)
    with tf.name_scope('C2H_image_comparison_summaries'):
        _add_comparison_summary(cyclegan_model.model_y2x, cyclegan_model.reconstructed_y)


def gan_model(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        # Real data and conditioning.
        real_data,
        generator_inputs,
        # Optional scopes.
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        # Options.
        check_shapes=True):
    """Returns GAN model outputs and variables.
  Args:
    generator_fn: A python lambda that takes `generator_inputs` as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
    real_data: A Tensor representing the real data.
    generator_inputs: A Tensor or list of Tensors to the generator. In the
      vanilla GAN case, this might be a single noise Tensor. In the conditional
      GAN case, this might be the generator's conditioning.
    generator_scope: Optional generator variable scope. Useful if you want to
      reuse a subgraph that has already been created.
    discriminator_scope: Optional discriminator variable scope. Useful if you
      want to reuse a subgraph that has already been created.
    check_shapes: If `True`, check that generator produces Tensors that are the
      same shape as real data. Otherwise, skip this check.
  Returns:
    A GANModel namedtuple.
  Raises:
    ValueError: If the generator outputs a Tensor that isn't the same shape as
      `real_data`.
  """
    # Create models
    with variable_scope.variable_scope(generator_scope) as gen_scope:
        generator_inputs = _convert_tensor_or_l_or_d(generator_inputs)
        generated_data = generator_fn(generator_inputs)
    with variable_scope.variable_scope(discriminator_scope) as dis_scope:
        discriminator_gen_outputs = discriminator_fn(generated_data, generator_inputs)
    with variable_scope.variable_scope(dis_scope, reuse=True):
        real_data = ops.convert_to_tensor(real_data)
        discriminator_real_outputs = discriminator_fn(real_data, generator_inputs)

    if check_shapes:
        if not generated_data.shape.is_compatible_with(real_data.shape):
            raise ValueError('Generator output shape (%s) must be the same shape as real data '
                             '(%s).' % (generated_data.shape, real_data.shape))

    # Get model-specific variables.
    generator_variables = variables_lib.get_trainable_variables(gen_scope)
    discriminator_variables = variables_lib.get_trainable_variables(dis_scope)

    return namedtuples.GANModel(generator_inputs, generated_data, generator_variables, gen_scope, generator_fn,
                                real_data, discriminator_real_outputs, discriminator_gen_outputs,
                                discriminator_variables, dis_scope, discriminator_fn)


def cyclegan_model(
        # Lambdas defining models.
        generator_fn,
        discriminator_fn,
        # data X and Y.
        images_healthy,
        mask_healthy,
        images_cancer,
        mask_cancer,
        # Optional scopes.
        generator_scope='Generator',
        discriminator_scope='Discriminator',
        model_x2y_scope='ModelH2C',
        model_y2x_scope='ModelC2H',
        # Options.
        check_shapes=True,
        include_masks=True):
    """Returns a CycleGAN model outputs and variables.
  See https://arxiv.org/abs/1703.10593 for more details.
  Args:
    generator_fn: A python lambda that takes `data_x` or `data_y` as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
    data_x: A `Tensor` of dataset X. Must be the same shape as `data_y`.
    data_y: A `Tensor` of dataset Y. Must be the same shape as `data_x`.
    generator_scope: Optional generator variable scope. Useful if you want to
      reuse a subgraph that has already been created. Defaults to 'Generator'.
    discriminator_scope: Optional discriminator variable scope. Useful if you
      want to reuse a subgraph that has already been created. Defaults to
      'Discriminator'.
    model_x2y_scope: Optional variable scope for model x2y variables. Defaults
      to 'ModelX2Y'.
    model_y2x_scope: Optional variable scope for model y2x variables. Defaults
      to 'ModelY2X'.
    check_shapes: If `True`, check that generator produces Tensors that are the
      same shape as `data_x` (`data_y`). Otherwise, skip this check.
  Returns:
    A `CycleGANModel` namedtuple.
  Raises:
    ValueError: If `check_shapes` is True and `data_x` or the generator output
      does not have the same shape as `data_y`.
  """

    # Create models.
    def _define_partial_model(input_data, output_data):
        return gan_model(
                generator_fn=generator_fn,
                discriminator_fn=discriminator_fn,
                real_data=output_data,
                generator_inputs=input_data,
                generator_scope=generator_scope,
                discriminator_scope=discriminator_scope,
                check_shapes=check_shapes)

    if include_masks:
        data_healthy = tf.stack([images_healthy, mask_healthy], axis=3)
        data_cancer = tf.stack([images_cancer, mask_cancer], axis=3)
    else:
        data_healthy = tf.expand_dims(images_healthy, axis=3)
        data_cancer = tf.expand_dims(images_cancer, axis=3)

    with variable_scope.variable_scope(model_x2y_scope):
        model_x2y = _define_partial_model(data_healthy, data_cancer)
    with variable_scope.variable_scope(model_y2x_scope):
        model_y2x = _define_partial_model(data_cancer, data_healthy)

    with variable_scope.variable_scope(model_y2x.generator_scope, reuse=True):
        reconstructed_x = model_y2x.generator_fn(model_x2y.generated_data)
    with variable_scope.variable_scope(model_x2y.generator_scope, reuse=True):
        reconstructed_y = model_x2y.generator_fn(model_y2x.generated_data)

    return namedtuples.CycleGANModel(model_x2y, model_y2x, reconstructed_x, reconstructed_y)


def _convert_tensor_or_l_or_d(tensor_or_l_or_d):
    """Convert input, list of inputs, or dictionary of inputs to Tensors."""
    if isinstance(tensor_or_l_or_d, (list, tuple)):
        return [ops.convert_to_tensor(x) for x in tensor_or_l_or_d]
    elif isinstance(tensor_or_l_or_d, dict):
        return {k: ops.convert_to_tensor(v) for k, v in tensor_or_l_or_d.items()}
    else:
        return ops.convert_to_tensor(tensor_or_l_or_d)


def cycle_consistency_loss(cyclegan_model, lambda_identity=0.0, scope=None, add_summaries=False):
    """Defines the cycle consistency loss.
    Uses `cycle_consistency_loss` to compute the cycle consistency loss for a
    `cyclegan_model`. Includes also identity loss.
    Args:
      cyclegan_model: A `CycleGANModel` namedtuple.
      scope: The scope for the operations performed in computing the loss.
        Defaults to None.
      add_summaries: Whether or not to add detailed summaries for the loss.
        Defaults to False.
    Returns:
      A scalar `Tensor` of cycle consistency loss.
    Raises:
      ValueError: If `cyclegan_model` is not a `CycleGANModel` namedtuple.
    """
    if not isinstance(cyclegan_model, namedtuples.CycleGANModel):
        raise ValueError('`cyclegan_model` must be a `CycleGANModel`. Instead, was %s.' % type(cyclegan_model))
    return cycle_consistency_loss_impl(cyclegan_model.model_x2y.generator_inputs,
                                       cyclegan_model.model_x2y.generated_data, cyclegan_model.reconstructed_x,
                                       cyclegan_model.model_y2x.generator_inputs,
                                       cyclegan_model.model_y2x.generated_data, cyclegan_model.reconstructed_y,
                                       lambda_identity, scope, add_summaries)


def cycle_consistency_loss_impl(data_x,
                                generated_y,
                                reconstructed_data_x,
                                data_y,
                                generated_x,
                                reconstructed_data_y,
                                lambda_identity=0.0,
                                scope=None,
                                add_summaries=False):
    """Defines the cycle consistency loss.
    The cyclegan model has two partial models where `model_x2y` generator F maps
    data set X to Y, `model_y2x` generator G maps data set Y to X. For a `data_x`
    in data set X, we could reconstruct it by
    * reconstructed_data_x = G(F(data_x))
    Similarly
    * reconstructed_data_y = F(G(data_y))
    The cycle consistency loss is about the difference between data and
    reconstructed data, namely
    * loss_x2x = |data_x - G(F(data_x))| (L1-norm)
    * loss_y2y = |data_y - F(G(data_y))| (L1-norm)
    * loss = (loss_x2x + loss_y2y) / 2
    where `loss` is the final result.
    See https://arxiv.org/abs/1703.10593 for more details.
    Args:
      data_x: A `Tensor` of data X.
      reconstructed_data_x: A `Tensor` of reconstructed data X.
      data_y: A `Tensor` of data Y.
      reconstructed_data_y: A `Tensor` of reconstructed data Y.
      scope: The scope for the operations performed in computing the loss.
        Defaults to None.
      add_summaries: Whether or not to add detailed summaries for the loss.
        Defaults to False.
    Returns:
      A scalar `Tensor` of cycle consistency loss.
    """

    def _partial_cycle_consistency_loss(data, reconstructed_data):
        # Following the original implementation
        # https://github.com/junyanz/CycleGAN/blob/master/models/cycle_gan_model.lua
        # use L1-norm of pixel-wise error normalized by data size so that
        # `cycle_loss_weight` can be specified independent of image size.
        return tf.reduce_mean(tf.abs(data - reconstructed_data))

    with ops.name_scope(
            scope, 'cycle_consistency_loss', values=[data_x, reconstructed_data_x, data_y, reconstructed_data_y]):
        loss_x2x = _partial_cycle_consistency_loss(data_x, reconstructed_data_x)
        loss_y2y = _partial_cycle_consistency_loss(data_y, reconstructed_data_y)
        loss_x2y = _partial_cycle_consistency_loss(data_x, generated_y)
        loss_y2x = _partial_cycle_consistency_loss(data_y, generated_x)
        loss = ((loss_x2x + loss_y2y) / 2.0) + lambda_identity * ((loss_x2y + loss_y2x) / 2.0)
        if add_summaries:
            tf.summary.scalar('cycle_consistency_loss_x2x', loss_x2x)
            tf.summary.scalar('cycle_consistency_loss_y2y', loss_y2y)
            tf.summary.scalar('cycle_consistency_loss_identity_x2y', loss_x2y)
            tf.summary.scalar('cycle_consistency_loss_identity_y2x', loss_y2x)
            tf.summary.scalar('cycle_consistency_loss', loss)

    return loss


# Least Squares loss from `Least Squares Generative Adversarial Networks`
# (https://arxiv.org/abs/1611.04076).
def hinge_generator_loss(discriminator_gen_outputs,
                         weights=1.0,
                         scope=None,
                         loss_collection=ops.GraphKeys.LOSSES,
                         reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                         add_summaries=False):
    """Hinge generator loss.
  Args:
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_label: The value that the generator is trying to get the discriminator
      to output on generated data.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_gen_outputs`, and must be broadcastable to
      `discriminator_gen_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
    with ops.name_scope(scope, 'hinge_generator_loss', (discriminator_gen_outputs, 1)) as scope:
        discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
        loss = -tf.reduce_mean(discriminator_gen_outputs)
        loss = losses.compute_weighted_loss(loss, weights, scope, loss_collection, reduction)

    if add_summaries:
        tf.summary.scalar('generator_lsq_loss', loss)

    return loss


def hinge_discriminator_loss(discriminator_real_outputs,
                             discriminator_gen_outputs,
                             real_weights=1.0,
                             generated_weights=1.0,
                             scope=None,
                             loss_collection=ops.GraphKeys.LOSSES,
                             reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
                             add_summaries=False):
    """Hinge discriminator loss.
  Args:
    discriminator_real_outputs: Discriminator output on real data.
    discriminator_gen_outputs: Discriminator output on generated data. Expected
      to be in the range of (-inf, inf).
    real_label: The value that the discriminator tries to output for real data.
    fake_label: The value that the discriminator tries to output for fake data.
    real_weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `discriminator_real_outputs`, and must be broadcastable to
      `discriminator_real_outputs` (i.e., all dimensions must be either `1`, or
      the same as the corresponding dimension).
    generated_weights: Same as `real_weights`, but for
      `discriminator_gen_outputs`.
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which this loss will be added.
    reduction: A `tf.losses.Reduction` to apply to loss.
    add_summaries: Whether or not to add summaries for the loss.
  Returns:
    A loss Tensor. The shape depends on `reduction`.
  """
    with ops.name_scope(scope, 'hinge_discriminator_loss', (discriminator_gen_outputs, 1)) as scope:
        print(discriminator_gen_outputs)
        discriminator_real_outputs = math_ops.to_float(discriminator_real_outputs)
        discriminator_gen_outputs = math_ops.to_float(discriminator_gen_outputs)
        discriminator_real_outputs.shape.assert_is_compatible_with(discriminator_gen_outputs.shape)

        real_losses = tf.reduce_mean(tf.nn.relu(1 - discriminator_real_outputs))
        fake_losses = tf.reduce_mean(tf.nn.relu(1 + discriminator_gen_outputs))

        loss_on_real = losses.compute_weighted_loss(
                real_losses, real_weights, scope, loss_collection=None, reduction=reduction)
        loss_on_generated = losses.compute_weighted_loss(
                fake_losses, generated_weights, scope, loss_collection=None, reduction=reduction)

        loss = loss_on_real + loss_on_generated
        util.add_loss(loss, loss_collection)

    if add_summaries:
        tf.summary.scalar('discriminator_gen_hinge_loss', loss_on_generated)
        tf.summary.scalar('discriminator_real_hinge_loss', loss_on_real)
        tf.summary.scalar('discriminator_hinge_loss', loss)

    return loss

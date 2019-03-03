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
"""Trains a CycleGAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from datetime import datetime
import os

import tensorflow as tf

import models.breast_cycle_gan.data_provider as data_provider
import models.breast_cycle_gan.discriminator as discriminator
import models.breast_cycle_gan.generator as generator
import models.breast_cycle_gan.custom.gan as mygan

from tensorflow.contrib.gan.python.losses.python.tuple_losses_impl import _args_to_gan_model

flags = tf.flags
tfgan = tf.contrib.gan

flags.DEFINE_enum("use_dataset", 'cbis', ['synth', 'cbis'], "Use synthetic dataset instead of real images.")

flags.DEFINE_enum("gan_type", 'lsgan', ['lsgan', 'hinge'], "Type of GAN loss.")

flags.DEFINE_bool("use_icnr", False, "Use kernel initialization as described in the Twitter paper.")

flags.DEFINE_bool("use_spectral_norm", False,
                  "Use spectral normalization in both discriminator and generator (https://arxiv.org/abs/1802.05957).")

flags.DEFINE_bool("use_self_attention", False,
                  "Use self attention for convolutions (https://arxiv.org/abs/1805.08318).")

flags.DEFINE_integer("num_resnet_blocks", 9, "Number of resnet blocks in the generator.")

flags.DEFINE_string('upsample_method', 'conv2d_transpose', 'Upsampling method for genrator.')

flags.DEFINE_string(
        'image_x_file',
        '/scratch_net/biwidl100/oskopek/transformed_wo_cbis_test_none/small_all_512x408_final/healthy.train.tfrecord',
        'File pattern of images in image set X')

flags.DEFINE_string(
        'image_y_file',
        '/scratch_net/biwidl100/oskopek/transformed_wo_cbis_test_none/small_all_512x408_final/cancer.train.tfrecord',
        'File pattern of images in image set Y')

flags.DEFINE_integer('batch_size', 1, 'The number of images in each batch.')

flags.DEFINE_integer('height', 512, 'The height of images.')
flags.DEFINE_integer('width', 408, 'The width of images. Must be a multiple of 4.')

flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')

flags.DEFINE_string('train_log_dir', None, 'Directory where to write event logs.')

flags.DEFINE_float('generator_lr', 0.0002, 'The compression model learning rate.')

flags.DEFINE_float('discriminator_lr', 0.0001, 'The discriminator learning rate.')

flags.DEFINE_integer('max_number_of_steps', 500000, 'The maximum number of gradient steps.')

flags.DEFINE_integer(
        'ps_tasks', 0, 'The number of parameter servers. If the value is 0, then the parameters '
        'are handled locally by the worker.')

flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training with multiple workers to '
                     'identify each worker.')

flags.DEFINE_float('cycle_consistency_loss_weight', 10.0, 'The weight of cycle consistency loss')

flags.DEFINE_integer(
        'checkpoint_hook_steps', -1, 'Number of steps between checkpoint hook activation. '
        'Negative numbers disable the checkpoint hook.')

flags.DEFINE_float('loss_identity_lambda', 0.5,
                   'The weight of cycle identity loss. Will be multiplied by cycle_consisteny_loss_weight.')

flags.DEFINE_bool('include_masks', True, "Is model conditioned on the ROIs.")

FLAGS = flags.FLAGS


def _define_model(images_healthy, images_cancer, include_masks=True):
    """Defines a CycleGAN model that maps between images_x and images_y.

    For our case, C=2.
    Args:
      images_x: A 4D float `Tensor` of NHWC format.  Images in set X. First channel image, second channel mask.
      images_y: A 4D float `Tensor` of NHWC format.  Images in set Y. First channel image, second channel mask.

    Returns:
      A `CycleGANModel` namedtuple.
    """
    print("FLAG: use icnr", FLAGS.use_icnr)
    print("FLAG: num_resnet_blocks:", FLAGS.num_resnet_blocks)
    print("FLAG: upsample_method:", FLAGS.upsample_method)
    num_outputs = 2 if include_masks else 1
    flagged_generator = functools.partial(
            generator.generator,
            num_resnet_blocks=FLAGS.num_resnet_blocks,
            use_icnr=FLAGS.use_icnr,
            upsample_method=FLAGS.upsample_method,
            num_outputs=num_outputs,
            use_spectral_norm=FLAGS.use_spectral_norm,
            self_attention=FLAGS.use_self_attention,
            is_training=True)
    flagged_discriminator = functools.partial(
            discriminator.discriminator,
            is_training=True,
            use_spectral_norm=FLAGS.use_spectral_norm,
            self_attention=FLAGS.use_self_attention)
    images_healthy, mask_healthy = tf.unstack(images_healthy, axis=3)
    images_cancer, mask_cancer = tf.unstack(images_cancer, axis=3)
    cyclegan_model = mygan.cyclegan_model(
            generator_fn=flagged_generator,
            discriminator_fn=flagged_discriminator,
            images_healthy=images_healthy,
            mask_healthy=mask_healthy,
            images_cancer=images_cancer,
            mask_cancer=mask_cancer,
            include_masks=include_masks)

    # Add summaries for generated images.
    mygan.add_cyclegan_image_summaries(cyclegan_model, include_masks)
    tfgan.eval.add_gan_model_summaries(cyclegan_model)

    return cyclegan_model


def _get_lr(base_lr):
    """Returns a learning rate `Tensor`.

  Args:
    base_lr: A scalar float `Tensor` or a Python number.  The base learning
        rate.

  Returns:
    A scalar float `Tensor` of learning rate which equals `base_lr` when the
    global training step is less than FLAGS.max_number_of_steps / 2, afterwards
    it linearly decays to zero.
  """
    global_step = tf.train.get_or_create_global_step()
    lr_constant_steps = FLAGS.max_number_of_steps // 2

    def _lr_decay():
        return tf.train.polynomial_decay(
                learning_rate=base_lr,
                global_step=(global_step - lr_constant_steps),
                decay_steps=(FLAGS.max_number_of_steps - lr_constant_steps),
                end_learning_rate=0.0)

    return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)


def _get_optimizer(gen_lr, dis_lr):
    """Returns generator optimizer and discriminator optimizer.

  Args:
    gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
        rate.
    dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
        learning rate.

  Returns:
    A tuple of generator optimizer and discriminator optimizer.
  """
    # beta1 follows
    # https://github.com/junyanz/CycleGAN/blob/master/options.lua
    gen_opt = tf.train.AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
    dis_opt = tf.train.AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
    return gen_opt, dis_opt


def _define_train_ops(cyclegan_model, cyclegan_loss):
    """Defines train ops that trains `cyclegan_model` with `cyclegan_loss`.

  Args:
    cyclegan_model: A `CycleGANModel` namedtuple.
    cyclegan_loss: A `CycleGANLoss` namedtuple containing all losses for
        `cyclegan_model`.

  Returns:
    A `GANTrainOps` namedtuple.
  """
    gen_lr = _get_lr(FLAGS.generator_lr)
    dis_lr = _get_lr(FLAGS.discriminator_lr)
    gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)
    train_ops = tfgan.gan_train_ops(
            cyclegan_model,
            cyclegan_loss,
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            summarize_gradients=True,
            colocate_gradients_with_ops=True,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    tf.summary.scalar('generator_lr', gen_lr)
    tf.summary.scalar('discriminator_lr', dis_lr)
    return train_ops


def main(_):
    if FLAGS.train_log_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        logdir = "{}/{}-{}".format('data_out', "CycleGanEst", timestamp)
    else:
        logdir = FLAGS.train_log_dir

    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)

    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        with tf.name_scope('inputs'):
            if FLAGS.use_dataset == 'synth':
                # Generated two channels. First channel image, second channel is mask.
                images_healthy, images_cancer = data_provider.provide_synth_dataset(
                        batch_size=FLAGS.batch_size, img_size=(FLAGS.height, FLAGS.width))
            elif FLAGS.use_dataset == 'cbis':
                # Generated two channels. First channel image, second channel is mask.
                images_healthy, images_cancer = data_provider.provide_cbis_dataset(
                        [FLAGS.image_x_file, FLAGS.image_y_file],
                        batch_size=FLAGS.batch_size,
                        img_size=(FLAGS.height, FLAGS.width))
            else:
                images_healthy, images_cancer = data_provider.provide_custom_datasets(
                        [FLAGS.image_set_x_file_pattern, FLAGS.image_set_y_file_pattern],
                        batch_size=FLAGS.batch_size,
                        img_size=(FLAGS.height, FLAGS.width))

        # Define CycleGAN model.
        print("images healthy", images_healthy.get_shape())
        print("images cancer", images_cancer.get_shape())
        cyclegan_model = _define_model(images_healthy, images_cancer, FLAGS.include_masks)

        # Define CycleGAN loss.
        if FLAGS.gan_type == 'lsgan':
            print("Using lsgan")
            generator_loss_fn = _args_to_gan_model(tfgan.losses.wargs.least_squares_generator_loss)
            discriminator_loss_fn = _args_to_gan_model(tfgan.losses.wargs.least_squares_discriminator_loss)
        elif FLAGS.gan_type == 'hinge':
            print("Using hinge")
            generator_loss_fn = _args_to_gan_model(mygan.hinge_generator_loss)
            discriminator_loss_fn = _args_to_gan_model(mygan.hinge_discriminator_loss)
        else:
            raise ValueError("Unknown gan type.")

        cyclegan_loss = tfgan.cyclegan_loss(
                cyclegan_model,
                cycle_consistency_loss_weight=FLAGS.cycle_consistency_loss_weight,
                generator_loss_fn=generator_loss_fn,
                discriminator_loss_fn=discriminator_loss_fn,
                cycle_consistency_loss_fn=functools.partial(
                        mygan.cycle_consistency_loss, lambda_identity=FLAGS.loss_identity_lambda),
                tensor_pool_fn=tfgan.features.tensor_pool)

        # Define CycleGAN train ops.
        train_ops = _define_train_ops(cyclegan_model, cyclegan_loss)

        # Training
        train_steps = tfgan.GANTrainSteps(1, 1)
        status_message = tf.string_join(
                ['Starting train step: ', tf.as_string(tf.train.get_or_create_global_step())], name='status_message')
        if not FLAGS.max_number_of_steps:
            return

        # To avoid problems with GPU memmory.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        # If a operation is not define it the default device, let it execute in another.
        config.allow_soft_placement = True
        hooks = [
                tf.train.StopAtStepHook(num_steps=FLAGS.max_number_of_steps),
                tf.train.LoggingTensorHook([status_message], every_n_iter=10),
        ]
        if FLAGS.checkpoint_hook_steps > 0:
            chkpt_hook = tf.train.CheckpointSaverHook(
                    checkpoint_dir=os.path.join(logdir, 'chook'),
                    save_steps=FLAGS.checkpoint_hook_steps,
                    saver=tf.train.Saver(max_to_keep=300))
            hooks.append(chkpt_hook)

        tfgan.gan_train(
                train_ops,
                logdir,
                hooks=hooks,
                get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
                master=FLAGS.master,
                is_chief=FLAGS.task == 0,
                config=config)


if __name__ == '__main__':
    # tf.flags.mark_flag_as_required('image_set_x_file_pattern')
    # tf.flags.mark_flag_as_required('image_set_y_file_pattern')
    tf.app.run()

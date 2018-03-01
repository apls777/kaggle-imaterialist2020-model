# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a DCGAN model on MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Standard Imports
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf

import cifar_input
import cifar_model
import mnist_input
import mnist_model
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.python.estimator import estimator

FLAGS = tf.flags.FLAGS

# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
tf.flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

# Model specific paramenters
tf.flags.DEFINE_string(
    'master', default=None,
    help='GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')
tf.flags.DEFINE_string('dataset', 'mnist',
                       'One of ["mnist", "cifar"]. Requires additional flags')
tf.flags.DEFINE_string('model_dir',
                       '', 'Output model directory')
tf.flags.DEFINE_integer('noise_dim',
                        64, 'Number of dimensions for the noise vector')
tf.flags.DEFINE_integer('batch_size',
                        1024, 'Batch size for both generator and discriminator')
tf.flags.DEFINE_integer('num_shards',
                        8, 'Number of TPU chips')
tf.flags.DEFINE_integer('train_steps',
                        1000, 'Number of training steps')
tf.flags.DEFINE_integer('train_steps_per_eval',
                        50, 'Steps per eval and image generation')
tf.flags.DEFINE_float('learning_rate', 0.0002,
                      'LR for both D and G')
tf.flags.DEFINE_boolean('eval_loss', False,
                        'Evaluate discriminator and generator loss during eval')
tf.flags.DEFINE_boolean('use_tpu',
                        True, 'Use TPU for training')

_NUM_VIZ_IMAGES = 100   # For generating a 10x10 grid of generator samples

# Global variables for data and model
dataset = None
model = None


def model_fn(features, labels, mode, params):
  """Constructs DCGAN from individual generator and discriminator networks."""
  del labels    # Unconditional GAN does not use labels

  if mode == tf.estimator.ModeKeys.PREDICT:
    ###########
    # PREDICT #
    ###########
    # Generate fixed random noise on device instead of feeding via input_fn
    np.random.seed(0)
    random_noise = tf.constant(
        np.random.randn(_NUM_VIZ_IMAGES, FLAGS.noise_dim), dtype=tf.float32)
    predictions = {
        'generated_images': model.generator(random_noise,
                                            is_training=False)
    }

    return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)

  random_noise = tf.random_normal([params['batch_size'], FLAGS.noise_dim])
  true_samples = features

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  generated_samples = model.generator(random_noise,
                                      is_training=is_training)

  # Get logits from discriminator
  d_on_data_logits = tf.squeeze(model.discriminator(true_samples))
  d_on_g_logits = tf.squeeze(model.discriminator(generated_samples))

  # Calculate discriminator loss
  d_loss_on_data = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_on_data_logits),
      logits=d_on_data_logits)
  d_loss_on_gen = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(d_on_g_logits),
      logits=d_on_g_logits)

  d_loss = d_loss_on_data + d_loss_on_gen

  # Calculate generator loss
  g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(d_on_g_logits),
      logits=d_on_g_logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    #########
    # TRAIN #
    #########
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)
    d_optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate, beta1=0.5)
    g_optimizer = tf.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate, beta1=0.5)

    if FLAGS.use_tpu:
      d_optimizer = tpu_optimizer.CrossShardOptimizer(d_optimizer)
      g_optimizer = tpu_optimizer.CrossShardOptimizer(g_optimizer)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      d_step = d_optimizer.minimize(
          d_loss,
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='Discriminator'))
      g_step = g_optimizer.minimize(
          g_loss,
          var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='Generator'))

      increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
      joint_op = tf.group([d_step, g_step, increment_step])

      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=g_loss,
          train_op=joint_op)

  elif mode == tf.estimator.ModeKeys.EVAL:
    ########
    # EVAL #
    ########
    def _eval_metric_fn(d_loss, g_loss):
      # When using TPUs, this function is run on a different machine than the
      # rest of the model_fn and should not capture any Tensors defined there
      return {
          'discriminator_loss': tf.metrics.mean(d_loss),
          'generator_loss': tf.metrics.mean(g_loss)}

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=tf.reduce_mean(g_loss),
        eval_metrics=(_eval_metric_fn, [d_loss, g_loss]))

  # Should never reach here
  raise ValueError('Invalid mode provided to model_fn')


def generate_input_fn(is_training):
  """Creates input_fn depending on whether the code is training or not."""
  return dataset.InputFunction(is_training)


def main(argv):
  del argv

  if FLAGS.master is None and FLAGS.tpu_name is None:
    raise RuntimeError('You must specify either --master or --tpu_name.')

  if FLAGS.master is not None:
    if FLAGS.tpu_name is not None:
      tf.logging.warn('Both --master and --tpu_name are set. Ignoring '
                      '--tpu_name and using --master.')
    tpu_grpc_url = FLAGS.master
  else:
    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      tpu_config=tpu_config.TPUConfig(
          num_shards=FLAGS.num_shards,
          iterations_per_loop=FLAGS.train_steps_per_eval))

  # Set module-level global variable so that model_fn and input_fn can be
  # identical for each different kind of dataset and model
  global dataset, model
  if FLAGS.dataset == 'mnist':
    dataset = mnist_input
    model = mnist_model
  elif FLAGS.dataset == 'cifar':
    dataset = cifar_input
    model = cifar_model
  else:
    raise ValueError('Invalid dataset: %s' % FLAGS.dataset)

  est = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

  current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)   # pylint: disable=protected-access,line-too-long
  tf.logging.info('Starting training for %d steps, current step: %d' %
                  (FLAGS.train_steps, current_step))
  while current_step < FLAGS.train_steps:
    next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                          FLAGS.train_steps)
    est.train(input_fn=generate_input_fn(True),
              max_steps=next_checkpoint)
    current_step = next_checkpoint
    tf.logging.info('Finished training step %d' % current_step)

    if FLAGS.eval_loss:
      # Evaluate loss on test set
      metrics = est.evaluate(input_fn=generate_input_fn(False),
                             steps=dataset.NUM_EVAL_IMAGES // FLAGS.batch_size)
      tf.logging.info('Finished evaluating')
      tf.logging.info(metrics)

    # Render some generated images
    # input_fn is a stub because noise is generated in the graph
    generated_iter = est.predict(input_fn=lambda params: (None, None))
    images = [generated_iter.next()['generated_images'][:, :, :]
              for _ in range(_NUM_VIZ_IMAGES)]
    image_rows = [np.concatenate(images[i:i+10], axis=0)
                  for i in range(0, _NUM_VIZ_IMAGES, 10)]
    tiled_image = np.concatenate(image_rows, axis=1)

    img = dataset.convert_array_to_image(tiled_image)

    step_string = str(current_step).zfill(5)
    file_obj = tf.gfile.Open(
        os.path.join(FLAGS.model_dir,
                     'generated_images', 'gen_%s.png' % (step_string)), 'w')
    img.save(file_obj, format='png')
    tf.logging.info('Finished generating images')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)

# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Main function to train various models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
from absl import flags
from absl import logging

from six.moves import range
import tensorflow_core._api.v1.compat.v1 as tf


from configs import factory
from dataloader import input_reader
from dataloader import mode_keys as ModeKeys
from executor import tpu_executor
from modeling import model_builder
from utils import config_utils

from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import params_dict

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()


flags.DEFINE_string(
    "mode", default="train", help="Mode to run: `train`, `eval` or `train_and_eval`."
)

flags.DEFINE_string(
    "model",
    default="retinanet",
    help="Support `retinanet`, `mask_rcnn`, `shapemask` and `classification`.",
)

flags.DEFINE_integer("num_cores", default=8, help="Number of TPU cores for training.")

flags.DEFINE_string(
    "tpu_job_name",
    None,
    "Name of TPU worker binary. Only necessary if job name is changed from"
    " default tpu_worker.",
)

flags.DEFINE_string("predict_file_pattern", None, "TFRecords file pattern.")
flags.DEFINE_string("predict_output_dir", None, "Output directory.")
flags.DEFINE_string(
    "predict_checkpoint_step", None, 'Checkpoint step for "predict" mode.'
)

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    params = factory.config_generator(FLAGS.model)

    if FLAGS.config_file:
        params = params_dict.override_params_dict(
            params, FLAGS.config_file, is_strict=True
        )

    params = params_dict.override_params_dict(
        params, FLAGS.params_override, is_strict=True
    )
    if not FLAGS.use_tpu:
        params.override(
            {
                "architecture": {
                    "use_bfloat16": False,
                },
                "batch_norm_activation": {
                    "use_sync_bn": False,
                },
            },
            is_strict=True,
        )
    params.override(
        {
            "platform": {
                "eval_master": FLAGS.eval_master,
                "tpu": FLAGS.tpu,
                "tpu_zone": FLAGS.tpu_zone,
                "gcp_project": FLAGS.gcp_project,
            },
            "tpu_job_name": FLAGS.tpu_job_name,
            "use_tpu": FLAGS.use_tpu,
            "model_dir": FLAGS.model_dir,
            "train": {
                "num_shards": FLAGS.num_cores,
            },
        },
        is_strict=False,
    )
    # Only run spatial partitioning in training mode.
    if FLAGS.mode != "train":
        params.train.input_partition_dims = None
        params.train.num_cores_per_replica = None

    params.validate()
    params.lock()
    pp = pprint.PrettyPrinter()
    params_str = pp.pformat(params.as_dict())
    logging.info("Model Parameters: %s", params_str)

    # Builds detection model on TPUs.
    model_fn = model_builder.ModelFn(params)
    executor = tpu_executor.TpuExecutor(model_fn, params)

    # Prepares input functions for train and eval.
    train_input_fn = input_reader.InputFn(
        params.train.train_file_pattern,
        params,
        mode=ModeKeys.TRAIN,
        dataset_type=params.train.train_dataset_type,
    )
    if params.eval.type == "customized":
        eval_input_fn = input_reader.InputFn(
            params.eval.eval_file_pattern,
            params,
            mode=ModeKeys.EVAL,
            dataset_type=params.eval.eval_dataset_type,
        )
    else:
        eval_input_fn = input_reader.InputFn(
            params.eval.eval_file_pattern,
            params,
            mode=ModeKeys.PREDICT_WITH_GT,
            dataset_type=params.eval.eval_dataset_type,
        )

    # Runs the model.
    if FLAGS.mode == "train":
        config_utils.save_config(params, params.model_dir)
        executor.train(train_input_fn, params.train.total_steps)
        if FLAGS.eval_after_training:
            executor.evaluate(
                eval_input_fn, params.eval.eval_samples // params.eval.eval_batch_size
            )

    elif FLAGS.mode == "eval":

        def terminate_eval():
            logging.info(
                "Terminating eval after %d seconds of no checkpoints",
                params.eval.eval_timeout,
            )
            return True

        # Runs evaluation when there's a new checkpoint.
        for ckpt in tf.train.checkpoints_iterator(
            params.model_dir,
            min_interval_secs=params.eval.min_eval_interval,
            timeout=params.eval.eval_timeout,
            timeout_fn=terminate_eval,
        ):
            # Terminates eval job when final checkpoint is reached.
            current_step = int(os.path.basename(ckpt).split("-")[1])

            logging.info("Starting to evaluate.")
            try:
                executor.evaluate(
                    eval_input_fn,
                    params.eval.eval_samples // params.eval.eval_batch_size,
                    ckpt,
                )

                if current_step >= params.train.total_steps:
                    logging.info(
                        "Evaluation finished after training step %d", current_step
                    )
                    break
            except tf.errors.NotFoundError:
                # Since the coordinator is on a different job than the TPU worker,
                # sometimes the TPU worker does not finish initializing until long after
                # the CPU job tells it to start evaluating. In this case, the checkpoint
                # file could have been deleted already.
                logging.info(
                    "Checkpoint %s no longer exists, skipping checkpoint", ckpt
                )

    elif FLAGS.mode == "train_and_eval":
        config_utils.save_config(params, params.model_dir)
        num_cycles = int(params.train.total_steps / params.eval.num_steps_per_eval)
        for cycle in range(num_cycles):
            logging.info("Start training cycle %d.", cycle)
            current_cycle_last_train_step = (cycle + 1) * params.eval.num_steps_per_eval
            executor.train(train_input_fn, current_cycle_last_train_step)
            executor.evaluate(
                eval_input_fn, params.eval.eval_samples // params.eval.eval_batch_size
            )

    elif FLAGS.mode == "predict":
        file_pattern = FLAGS.predict_file_pattern
        if not file_pattern:
            raise ValueError('"predict_file_pattern" parameter is required.')

        output_dir = FLAGS.predict_output_dir
        if not output_dir:
            raise ValueError('"predict_output_dir" parameter is required.')

        test_input_fn = input_reader.InputFn(
            file_pattern,
            params,
            mode=ModeKeys.PREDICT_WITH_GT,
            dataset_type=params.eval.eval_dataset_type,
        )

        checkpoint_prefix = "model.ckpt-" + FLAGS.predict_checkpoint_step
        checkpoint_path = os.path.join(FLAGS.model_dir, checkpoint_prefix)
        if not tf.train.checkpoint_exists(checkpoint_path):
            checkpoint_path = os.path.join(
                FLAGS.model_dir, "best_checkpoints", checkpoint_prefix
            )
            if not tf.train.checkpoint_exists(checkpoint_path):
                raise ValueError(
                    "Checkpoint not found: %s/%s" % (FLAGS.model_dir, checkpoint_prefix)
                )

        executor.predict(test_input_fn, checkpoint_path, output_dir=output_dir)

    else:
        logging.info("Mode not found.")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    tf.app.run(main)

import numpy as np
from tensorpack.utils.timer import timed_operation
from tensorpack.utils import logger

import tensorflow as tf
from resources.data.features import example_to_str, example_to_numpy, example_to_int
from random import shuffle
from itertools import repeat

CLASS_NAMES = ['BACKGROUND', 'BENIGN', 'MALIGNANT']


class BreastDetection(object):

    def __init__(self, tfrecords_pattern):
        self.tfrecords_pattern = tfrecords_pattern

    def get_examples(self,
                     tfrecords_glob,
                     options=tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)):
        for file in tf.gfile.Glob(tfrecords_glob):
            for record in tf.python_io.tf_record_iterator(file, options=options):
                example = tf.train.Example()
                example.ParseFromString(record)
                yield example

    def load(self, add_gt=True):
        with timed_operation("Load groundtruth boxes from {}".format(self.tfrecords_pattern)):
            examples = self.get_examples(self.tfrecords_pattern)
            result = []
            skipped = 0
            for i, example in enumerate(examples):
                image_path = example_to_str(example, 'image_path')
                old_path = example_to_str(example, 'path')
                w = example_to_int(example, 'width')
                h = example_to_int(example, 'height')
                bboxes = example_to_numpy(example, 'bboxes', np.float32, (-1, 4))
                if bboxes.shape[0] == 0:
                    skipped += 1
                assert len(bboxes.shape) == 2
                assert bboxes.shape[1] == 4
                lbl = example_to_int(example, 'label') + 1  # BACKGROUND is 0
                labels = np.asarray(list(repeat(lbl, len(bboxes))))
                is_crowd = np.zeros(len(bboxes))
                result.append({
                        'height': h,
                        'width': w,
                        'id': old_path,
                        'file_name': image_path,
                        'boxes': bboxes,
                        'class': labels,
                        'is_crowd': is_crowd,
                        'label': lbl - 1
                })
            logger.info('{} images have zero boxes.'.format(skipped))
            return result

    def load_many(tfrecords_pattern, shuffle_dataset=True):
        c = BreastDetection(tfrecords_pattern)
        metadata = c.load()
        logger.info("#images loaded: {}".format(len(metadata)))
        if shuffle_dataset:
            shuffle(metadata)
            logger.info("#images shuffled: {}".format(len(metadata)))
        return metadata

# Copyright 2016 Google Inc. All Rights Reserved.
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

r"""CelebA dataset formating.

Download img_align_celeba.zip from
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html under the
link "Align&Cropped Images" in the "Img" directory and list_eval_partition.txt
under the link "Train/Val/Test Partitions" in the "Eval" directory. Then do:
unzip img_align_celeba.zip

Use the script as follow:
python celeba_formatting.py \
    --partition_fn [PARTITION_FILE_PATH] \
    --file_out [OUTPUT_FILE_PATH_PREFIX] \
    --fn_root [CELEBA_FOLDER] \
    --set [SUBSET_INDEX]

"""

import os
import os.path

import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import tensorflow as tf

from getAttributes import Attributes

# Currently set to default for train
tf.flags.DEFINE_string("file_out", "./datasets/celebA/celeba_train",
                       "Filename of the output .tfrecords file.")
tf.flags.DEFINE_string("fn_root", "./datasets/celebA/images", "Name of root file path.")
tf.flags.DEFINE_string("partition_fn", "./datasets/celebA/list_eval_partition.txt", "Partition file path.")
tf.flags.DEFINE_string("set", "0", "Name of subset.")
tf.flags.DEFINE_string("attr_filename", "./datasets/celebA/list_attr_celeba.txt", "Attributes file path.")

FLAGS = tf.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def celeba_format():
    """Main converter function."""
    # Celeb A
    with open(FLAGS.partition_fn, "r") as infile:
        img_fn_list = infile.readlines()
    img_fn_list = [elem.strip().split() for elem in img_fn_list]
    img_fn_list = [elem[0] for elem in img_fn_list if elem[1] == FLAGS.set]
    fn_root = FLAGS.fn_root
    num_examples = len(img_fn_list)

    file_out = "%s.tfrecords" % FLAGS.file_out
    writer = tf.python_io.TFRecordWriter(file_out)

    # use `attr` to index into attributes of each file
    attr = Attributes(inputfile=FLAGS.attr_filename).attributeMap

    for example_idx, img_fn in enumerate(img_fn_list):
        if example_idx % 1000 == 0:
            print (example_idx, "/", num_examples)
        image_raw = scipy.ndimage.imread(os.path.join(fn_root, img_fn))
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        image_raw = image_raw.tostring()
        label_male = 1 #  max(0, attr[img_fn]['Male'])

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int64_feature(rows),
                    "width": _int64_feature(cols),
                    "depth": _int64_feature(depth),
                    "image_raw": _bytes_feature(image_raw),
                    'label_male': _int64_feature(int(label_male))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()

def num_examples(tf_record_filename): 
    c = 0
    for record in tf.python_io.tf_record_iterator(tf_record_filename):
        c += 1
    return c

if __name__ == "__main__":
#    celeba_format()
    print(num_examples(FLAGS.file_out+'.tfrecords'))

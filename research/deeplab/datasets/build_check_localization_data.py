import math
import os.path
import sys
import build_data
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './check_localization',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './check_localization',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './check_localization/metadata',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './check_localization/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

_NUM_SHARDS = 32

def resize_image(img, size=300):
    '''
    Resize images as keeping aspect ratio.
    '''
    shape = img.shape[:2]
    channel = img.shape[2]
    ratio = np.min(shape) / float(size)
    new_shape = list((np.array(shape) / ratio).astype('int'))

    img = tf.expand_dims(img, 0) ## [1, width, height, channel]
    if channel == 3:
        img = tf.image.resize_images(img, size=new_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    elif channel == 1:
        img = tf.image.resize_images(img, size=new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
    img = tf.squeeze(img, 0)
    return img

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n').split(',') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

#  print("n images:", num_images)
#  print("n per shard:", num_per_shard)
#  print(filenames[0])

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(FLAGS.output_dir, '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    if os.path.isfile(output_filename):
        continue

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)

      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))
        sys.stdout.flush()

        # Read the image data
        image_filename = os.path.join(FLAGS.image_folder, filenames[i][0])
        with tf.gfile.FastGFile(image_filename, 'rb') as f:
            image_data = f.read()
        #image_data = tf.gfile.FastGFile(image_filename, 'rb').read()

        ## Resize image
        image = image_reader.decode_image(image_data)
        image = resize_image(image)
        height, width = image.shape[:2]
        image_data = image_reader.encode_image(image)

#        height, width = image_reader.read_image_dims(image_data)

        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(FLAGS.semantic_segmentation_folder, filenames[i][1])
        with tf.gfile.FastGFile(seg_filename, 'rb') as f:
            seg_data = f.read()
        #seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()

        ## Resize mask
        seg_image = label_reader.decode_image(seg_data)
        seg_image = resize_image(seg_image)
        seg_height, seg_width = seg_image.shape[:2]
        seg_data = label_reader.encode_image(seg_image)

#        seg_height, seg_width = label_reader.read_image_dims(seg_data, resize=True)

        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(image_data, filenames[i][0], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

    del example, tfrecord_writer, 

def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)

if __name__ == '__main__':
  tf.app.run()

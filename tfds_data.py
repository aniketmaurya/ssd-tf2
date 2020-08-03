import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

from box_utils import compute_target
from image_utils import horizontal_flip_tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def draw_boxes(image: np.ndarray, boxes: np.ndarray):
    pil_image = Image.fromarray(image)
    h, w = pil_image.size
    patch = ImageDraw.Draw(pil_image)

    for box in boxes:
        xmin, ymin, xmax, ymax = box * (h, w, h, w)

        patch.rectangle((xmin, ymin, xmax, ymax))

    return pil_image


class Dataset:
    """ Class for TFDS Dataset

    Attributes:
        data_dir: dataset data dir (ex: '/data/tensorflow_datasets')
    """

    def __init__(self, default_boxes,
                 new_size, mode='train', augmentation=None, data_dir='/data/tensorflow_datasets'):
        super(Dataset, self).__init__()
        self.idx_to_name = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']
        self.name_to_idx = dict([(v, k)
                                 for k, v in enumerate(self.idx_to_name)])
        self.augmentation = augmentation
        self.mode = mode
        self.new_size = new_size
        self.default_boxes = default_boxes

        data, info = tfds.load(
            'voc', split=mode,
            data_dir=data_dir,
            shuffle_files=True, with_info=True)

        self.info = info
        self.data = data.map(self.clean_data, AUTOTUNE)
    
    @tf.function
    def augmentation(self, image, boxes, labels):
        if tf.random.uniform(()) > 0.5:
            image, boxes, labels = horizontal_flip_tf(image, boxes, labels)
        
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_jpeg_quality(image)
        
        return image, boxes, labels

    @tf.function
    def image_preprocessing(self, filename, image, boxes, labels):
        if self.mode == 'train':
            image, boxes, labels = self.augmentation(image, boxes, labels)

        image = tf.image.resize(image, (self.new_size, self.new_size))
        image = (image / 127.0) - 1.0

        return filename, image, boxes, labels

    @tf.function
    def clean_data(self, data):
        image = data['image']
        filename = data['image/filename']
        labels = data['objects']['label']
        bbox = data['objects']['bbox']
        xy_bbox = tf.concat((bbox[..., -3::-1], bbox[..., -1:-3:-1]), 1)
        
        return filename, image, xy_bbox, labels

    @tf.function
    def map_compute_target(self, filename, image, bbox, labels):
        tout = [tf.int64, tf.float32]
        gt_confs, gt_locs = tf.py_function(compute_target, [self.default_boxes, bbox, labels], tout)
            
        return filename, image, gt_confs, gt_locs

    def generate(self):
        """
        Returns:
            filename: filename of image
            img: tensor of shape (300, 300, 3)
            boxes: tensor of shape (num_gt, 4)
            labels: tensor of shape (num_gt,)
        """
        data = self.data
        data = data.map(self.image_preprocessing, AUTOTUNE)
        data = data.map(self.map_compute_target, AUTOTUNE)
        return data


def create_batch_generator(default_boxes, new_size, batch_size, num_batches=None, prefetch=True, data_dir='/data/tensorflow_datasets'):
    train_dataset = Dataset(default_boxes,
                            new_size=new_size, mode='train', data_dir=data_dir)

    val_dataset = Dataset(default_boxes,
                          new_size=new_size, mode='validation', data_dir=data_dir)
    info = {
        'idx_to_name': train_dataset.idx_to_name,
        'name_to_idx': train_dataset.name_to_idx,
        'length': train_dataset.info.splits['train'].num_examples,
        'val_length': val_dataset.info.splits['validation'].num_examples,
        # 'image_dir': dataset.image_dir,
        # 'anno_dir': dataset.anno_dir
    }

    data_gen = train_dataset.generate()
    val_gen = val_dataset.generate()

    if batch_size:
        data_gen = data_gen.batch(batch_size)
        val_gen = val_gen.batch(batch_size)
    if num_batches:
        data_gen = data_gen.take(num_batches)

    if prefetch:
        print('data generator is prefetched...')
        data_gen = data_gen.prefetch(AUTOTUNE)
        val_gen = val_gen.prefetch(AUTOTUNE)

    print(f'Generated data with batch-size:{batch_size} and num-batches:{num_batches}')
    print(f'{info}')

    return data_gen, val_gen, info

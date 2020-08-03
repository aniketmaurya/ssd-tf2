import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

from box_utils import compute_target
from image_utils import horizontal_flip_tf


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
        self.new_size = new_size
        self.default_boxes = default_boxes

        data, info = tfds.load(
            'voc', split=mode,
            data_dir=data_dir,
            shuffle_files=True, with_info=True)

        self.info = info
        self.data = data.map(self.clean_data)
        # self.data = self.scaleup_bbox(self.data)

    
    @tf.function
    def preprocessing(self, filename, image, boxes, labels):
        if tf.random.uniform(()) > 0.5:
            image, boxes, labels = horizontal_flip_tf(image, boxes, labels)

        image = tf.image.resize(image, (self.new_size, self.new_size))
        image = (image/127.0) - 1.0
        
        return filename, image, boxes, labels

    @tf.function
    def clean_data(self, data):
        image = tf.image.resize(data['image'], (self.new_size, self.new_size))
        image = (image/127.0) - 1.0
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
        data = data.map(self.preprocessing)
        data = data.map(self.map_compute_target)
        return data


def create_batch_generator(default_boxes,
                           new_size, batch_size, num_batches=None,
                           augmentation=True, data_dir='/data/tensorflow_datasets'):
    train_dataset = Dataset(default_boxes,
                            new_size=new_size, mode='train', data_dir=data_dir, augmentation=augmentation)

    val_dataset = Dataset(default_boxes,
                          new_size=new_size, mode='validation', data_dir=data_dir)
    info = {
        'idx_to_name': train_dataset.idx_to_name,
        'name_to_idx': train_dataset.name_to_idx,
        'length': train_dataset.info.splits['train'].num_examples,
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
    print(f'Generated data with batch-size:{batch_size} and num-batches:{num_batches}')
    print(f'{info}')
    return data_gen, val_gen, info

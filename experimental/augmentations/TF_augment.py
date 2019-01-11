import tensorflow as tf
import glob
import time
import jpeg4py as jpeg
from time import time

class TFAug:
    def __init__(self, image_Dir = "/home/vladimir/work/test_image/vehicle/*.jpg", batch_Size = 64, prefetch_size = 8, Iterations = 1000):
        self.image_dir = image_Dir
        self.batch_size = batch_Size
        self.prefetch_size = prefetch_size
        self.iterations = Iterations
        self.images = []
        self.labels = []
        self.all_tf_flip_time = []
        self.all_tf_brightness_time = []
        self.all_tf_contrast_time = []
        self.all_tf_saturation_time = []

    def create_dataset(self):
        cv_images = [jpeg.JPEG(file).decode()[..., ::-1] for file in
                     glob.glob(self.image_dir)]

        def generator():
            for img in cv_images:
                yield img

        return tf.data.Dataset.from_generator(generator, tf.uint8, tf.TensorShape([450, 800, 3]))

    def input_fn(self):
        dataset = self.create_dataset()
        dataset = dataset.repeat().batch(self.batch_size).prefetch(self.prefetch_size)

        image = dataset.make_one_shot_iterator().get_next()
        return image

    def flip(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        image = self.input_fn()
        # flip_res = tf.image.flip_left_right(image)

        with tf.Session(config=config) as sess:
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([image])
            self.all_tf_flip_time.append(self.batch_size*self.iterations / (time() - tmp))

    def brightness(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        image = self.input_fn()
        brightness_res = tf.image.adjust_brightness(image, delta=0.5)

        with tf.Session(config=config) as sess:
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([brightness_res])
            self.all_tf_brightness_time.append(self.batch_size*self.iterations / (time() - tmp))

    def contrast(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        image = self.input_fn()
        contrast_res = tf.image.adjust_contrast(image, contrast_factor=1.5)

        with tf.Session(config=config) as sess:
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([contrast_res])
            self.all_tf_contrast_time.append(self.batch_size*self.iterations / (time() - tmp))

    def saturation(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        image = self.input_fn()
        saturation_res = tf.image.adjust_saturation(image, saturation_factor=0.2)

        with tf.Session(config=config) as sess:
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([saturation_res])
            self.all_tf_saturation_time.append(self.batch_size*self.iterations / (time() - tmp))

    def run(self):
        self.flip()
        self.brightness()
        self.contrast()
        self.saturation()

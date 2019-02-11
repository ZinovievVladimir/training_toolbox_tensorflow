import tensorflow as tf
import cv2
import time
import numpy as np
from time import time

class OpenCVAug:
    def __init__(self, image_dir, dim, batch_size = 64,
                 Iterations = 1000):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.iterations = Iterations
        self.images = []
        self.labels = []
        self.all_img_per_sec = []
        self.dim = dim

        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(self.image_dir))

        # Read an entire image file which is required since they're JPEGs, if the images
        # are too large they could be split in advance to smaller files or use the Fixed
        # reader to split up the file.
        image_reader = tf.WholeFileReader()

        # Read a whole file from the queue, the first returned value in the tuple is the
        # filename which we are ignoring.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tensor which we can
        # then use in training.
        self.image = tf.image.decode_jpeg(image_file, channels=3)
        #self.resized_img = tf.image.resize_images(self.image, self.dim)
        # Generate batch
        self.num_preprocess_threads = 4
        self.min_queue_examples = 256

    def resize(self, img):
        return cv2.resize(img, self.dim)

    def cv_flip(self, img):
        return cv2.flip(img, 1)

    def cv_brightness(self, img):
        delta_mat = np.full_like(img, 100)
        return cv2.subtract(img, delta_mat)

    def cv_contrast(self, img):
        return cv2.addWeighted(img, 1.5, 0, 0, 0)

    def cv_saturation(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.addWeighted(hsv[:, :, 1], 0.2, 0, 0, 0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def flip(self):
        config = tf.ConfigProto()
        resized_img = tf.py_func(self.resize, [self.image], (tf.uint8,))
        flip_res = tf.py_func(self.cv_flip, [resized_img[0]], (tf.uint8,))
        flip_res[0].set_shape((self.dim[0], self.dim[1], 3))
        images = tf.train.shuffle_batch(
            [flip_res],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=self.min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=self.min_queue_examples)
        # flip_res = tf.py_func(self.cv_flip, [self.images], (tf.uint8,))
        with tf.Session(config=config) as sess:
            # Required to get the filename matching to run.
            tf.local_variables_initializer().run()

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([images])
            self.all_img_per_sec.append(("CV_flip", self.batch_size*self.iterations / (time() - tmp)))

    def brightness(self):
        config = tf.ConfigProto()
        resized_img = tf.py_func(self.resize, [self.image], (tf.uint8,))
        brightness_res = tf.py_func(self.cv_brightness, [resized_img[0]], (tf.uint8,))
        brightness_res[0].set_shape((self.dim[0], self.dim[1], 3))
        images = tf.train.shuffle_batch(
            [brightness_res],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=self.min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=self.min_queue_examples)
        with tf.Session(config=config) as sess:
            # Required to get the filename matching to run.
            tf.local_variables_initializer().run()

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([images])
            self.all_img_per_sec.append(("CV_brightness", self.batch_size*self.iterations / (time() - tmp)))

    def contrast(self):
        config = tf.ConfigProto()
        resized_img = tf.py_func(self.resize, [self.image], (tf.uint8,))
        contrast_res = tf.py_func(self.cv_contrast, [resized_img[0]], (tf.uint8,))
        contrast_res[0].set_shape((self.dim[0], self.dim[1], 3))
        images = tf.train.shuffle_batch(
            [contrast_res],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=self.min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=self.min_queue_examples)
        with tf.Session(config=config) as sess:
            # Required to get the filename matching to run.
            tf.local_variables_initializer().run()

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([images])
            self.all_img_per_sec.append(("CV_contrast", self.batch_size*self.iterations / (time() - tmp)))

    def saturation(self):
        config = tf.ConfigProto()
        resized_img = tf.py_func(self.resize, [self.image], (tf.uint8,))
        saturation_res = tf.py_func(self.cv_saturation, [resized_img[0]], (tf.uint8,))
        saturation_res[0].set_shape((self.dim[0], self.dim[1], 3))
        images = tf.train.shuffle_batch(
            [saturation_res],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=self.min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=self.min_queue_examples)
        with tf.Session(config=config) as sess:
            # Required to get the filename matching to run.
            tf.local_variables_initializer().run()

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([images])
            self.all_img_per_sec.append(("CV_saturation", self.batch_size*self.iterations / (time() - tmp)))

    def run(self, flip, bri, con, sat):
        if flip:
            self.flip()
        if bri:
            self.brightness()
        if con:
            self.contrast()
        if sat:
            self.saturation()

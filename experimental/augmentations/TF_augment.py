import tensorflow as tf
import time
from time import time


class TFAug:
    def __init__(self, image_dir, dim, batch_Size = 64,
                 Iterations = 1000):
        self.image_dir = image_dir
        self.batch_size = batch_Size
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
        self.resized_img = tf.image.resize_images(self.image, self.dim)

        # Generate batch
        self.num_preprocess_threads = 4
        self.min_queue_examples = 256

    # def empty(self):
    #     config = tf.ConfigProto()
    #
    #     flip_res = tf.identity(self.resized_img)
    #
    #     flip_res.set_shape((self.dim[0], self.dim[1], 3))
    #     images = tf.train.shuffle_batch(
    #         [flip_res],
    #         batch_size=self.batch_size,
    #         num_threads=self.num_preprocess_threads,
    #         capacity=self.min_queue_examples + 3 * self.batch_size,
    #         min_after_dequeue=self.min_queue_examples)
    #
    #     # flip_res = tf.image.flip_left_right(self.images)
    #     with tf.Session(config=config) as sess:
    #         # Required to get the filename matching to run.
    #         tf.local_variables_initializer().run()
    #
    #         # Coordinate the loading of image files.
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)
    #
    #         tmp = time()
    #         for _ in range(self.iterations):
    #             res = sess.run([images])
    #         self.all_img_per_sec.append(("TF_empty", self.batch_size * self.iterations / (time() - tmp)))

    def flip(self):
        config = tf.ConfigProto()

        flip_res = tf.image.flip_left_right(self.resized_img)

        flip_res.set_shape((self.dim[0], self.dim[1], 3))
        images = tf.train.shuffle_batch(
            [flip_res],
            batch_size=self.batch_size,
            num_threads=self.num_preprocess_threads,
            capacity=self.min_queue_examples + 3 * self.batch_size,
            min_after_dequeue=self.min_queue_examples)

        # flip_res = tf.image.flip_left_right(self.images)
        with tf.Session(config=config) as sess:
            # Required to get the filename matching to run.
            tf.local_variables_initializer().run()

            # Coordinate the loading of image files.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            tmp = time()
            for _ in range(self.iterations):
                res = sess.run([images])
            self.all_img_per_sec.append(("TF_flip", self.batch_size*self.iterations / (time() - tmp)))

    def brightness(self):
        config = tf.ConfigProto()

        brightness_res = tf.image.adjust_brightness(self.resized_img, delta=0.5)
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
            self.all_img_per_sec.append(("TF_brightness", self.batch_size*self.iterations / (time() - tmp)))

    def contrast(self):
        config = tf.ConfigProto()

        contrast_res = tf.image.adjust_contrast(self.resized_img, contrast_factor=1.5)
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
            self.all_img_per_sec.append(("TF_contrast", self.batch_size*self.iterations / (time() - tmp)))

    def saturation(self):
        config = tf.ConfigProto()

        saturation_res = tf.image.adjust_saturation(self.resized_img, saturation_factor=0.2)
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
            self.all_img_per_sec.append(("TF_saturation", self.batch_size*self.iterations / (time() - tmp)))

    def run(self, flip, bri, con, sat):
        # self.empty()
        if flip:
            self.flip()
        if bri:
            self.brightness()
        if con:
            self.contrast()
        if sat:
            self.saturation()

import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from time import time

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir = "/home/vladimir/work/test_image"):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.flip = ops.Flip(device="gpu", vertical=0, horizontal=1)
        self.brightness = ops.Brightness(device="gpu", brightness=0.5)
        self.contrast = ops.Contrast(device="gpu", contrast=1.5)
        self.saturation = ops.Saturation(device="gpu", saturation=0.2)
        self.input = ops.FileReader(file_root=image_dir)


class FileReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images.gpu(), labels.gpu())


class FlipPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(FlipPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.flip(images.gpu())
        return (images, labels.gpu())


class BrightnessPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(BrightnessPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.brightness(images.gpu())
        return (images, labels.gpu())


class ContrastPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(ContrastPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.contrast(images.gpu())
        return (images, labels.gpu())


class SaturationPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SaturationPipeline, self).__init__(batch_size, num_threads, device_id)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.saturation(images.gpu())
        return (images, labels.gpu())



class DALIAug:
    def __init__(self, image_Dir = "/home/vladimir/work/test_image", batch_Size = 64, Iterations = 1000):
        self.image_dir = image_Dir
        self.batch_size = batch_Size
        self.iterations = Iterations
        self.images = []
        self.labels = []
        self.all_img_per_sec = []

        pipes = [FlipPipeline(batch_size=self.batch_size, num_threads=2, device_id=0),
                 BrightnessPipeline(batch_size=self.batch_size, num_threads=2, device_id=0),
                 ContrastPipeline(batch_size=self.batch_size, num_threads=2, device_id=0),
                 SaturationPipeline(batch_size=self.batch_size, num_threads=2, device_id=0)]
        s_pipes = [pipe.serialize() for pipe in pipes]

        daliop = dali_tf.DALIIterator()

        self.images = []
        self.labels = []

        for pipe in s_pipes:
            image, label = daliop(serialized_pipeline=pipe,
                                  shape=(self.batch_size, 450, 800, 3),
                                  image_type=tf.int32,
                                  label_type=tf.float32)
            self.images.append(image)
            self.labels.append(label)

    def flip(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[0], self.labels[0]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(pipe_img_per_sec)

    def brightness(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[1], self.labels[1]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(pipe_img_per_sec)

    def contrast(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[2], self.labels[2]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(pipe_img_per_sec)

    def saturation(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[3], self.labels[3]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(pipe_img_per_sec)

    def run(self):
        self.flip()
        self.brightness()
        self.contrast()
        self.saturation()

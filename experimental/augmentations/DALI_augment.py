import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from time import time


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, dim, image_dir):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_x=dim[0], resize_y=dim[1])
        self.flip = ops.Flip(device="gpu", vertical=0, horizontal=1)
        self.brightness = ops.Brightness(device="gpu", brightness=0.5)
        self.contrast = ops.Contrast(device="gpu", contrast=1.5)
        self.saturation = ops.Saturation(device="gpu", saturation=0.2)
        self.input = ops.FileReader(file_root=image_dir)
        self.uniform = ops.Uniform(range=(0.0, 1.0))


class FileReadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, dim = (256,256)):
        super(FileReadPipeline, self).__init__(batch_size, num_threads, device_id, image_dir=image_dir, dim = dim)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images.gpu(), labels.gpu())


class FlipPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, dim = (256,256)):
        super(FlipPipeline, self).__init__(batch_size, num_threads, device_id, image_dir=image_dir, dim = dim)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.flip(images.gpu())
        images = self.resize(images)
        return (images, labels.gpu())


class BrightnessPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, dim = (256,256)):
        super(BrightnessPipeline, self).__init__(batch_size, num_threads, device_id, image_dir=image_dir, dim = dim)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.brightness(images.gpu())
        images = self.resize(images)
        return (images, labels.gpu())


class ContrastPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, dim = (256,256)):
        super(ContrastPipeline, self).__init__(batch_size, num_threads, device_id, image_dir=image_dir, dim = dim)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.contrast(images.gpu())
        images = self.resize(images)
        return (images, labels.gpu())


class SaturationPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, image_dir, dim = (256,256)):
        super(SaturationPipeline, self).__init__(batch_size, num_threads, device_id, image_dir=image_dir, dim = dim)

    def define_graph(self):
        images, labels = self.input()
        images = self.decode(images)
        images = self.saturation(images.gpu())
        images = self.resize(images)
        return (images, labels.gpu())


class DALIAug:
    def __init__(self, image_dir, batch_Size=64,
                 Iterations=1000, dim=(256, 256)):
        self.image_dir = image_dir
        self.batch_size = batch_Size
        self.iterations = Iterations
        self.images = []
        self.labels = []
        self.all_img_per_sec = []

        pipes = [FlipPipeline(batch_size=self.batch_size, num_threads=2, device_id=0, dim = dim, image_dir=image_dir),
                 BrightnessPipeline(batch_size=self.batch_size, num_threads=2, device_id=0, dim = dim, image_dir=image_dir),
                 ContrastPipeline(batch_size=self.batch_size, num_threads=2, device_id=0, dim = dim, image_dir=image_dir),
                 SaturationPipeline(batch_size=self.batch_size, num_threads=2, device_id=0, dim = dim, image_dir=image_dir)]
        s_pipes = [pipe.serialize() for pipe in pipes]

        daliop = dali_tf.DALIIterator()

        self.images = []
        self.labels = []

        for pipe in s_pipes:
            image, label = daliop(serialized_pipeline=pipe,
                                  shapes = [(self.batch_size, dim[0], dim[1], 3), ()],
                                  dtypes = [tf.int32, tf.float32])
            self.images.append(image)
            self.labels.append(label)

    def flip(self):
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[0], self.labels[0]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(("DALI_flip", sum(pipe_img_per_sec)/ len(pipe_img_per_sec)))

    def brightness(self):
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[1], self.labels[1]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(("DALI_brightness", sum(pipe_img_per_sec)/ len(pipe_img_per_sec)))

    def contrast(self):
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[2], self.labels[2]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(("DALI_contrast", sum(pipe_img_per_sec)/ len(pipe_img_per_sec)))

    def saturation(self):
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            pipe_img_per_sec = []
            for i in range(self.iterations):
                start_time = time()

                res = sess.run([self.images[3], self.labels[3]])

                elapsed_time = time() - start_time
                img_per_sec = self.batch_size / elapsed_time
                pipe_img_per_sec.append(img_per_sec)
            self.all_img_per_sec.append(("DALI_saturation", sum(pipe_img_per_sec)/ len(pipe_img_per_sec)))

    def run(self, flip, bri, con, sat):
        if flip:
            self.flip()
        if bri:
            self.brightness()
        if con:
            self.contrast()
        if sat:
            self.saturation()

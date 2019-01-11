from DALI_augment import DALIAug
from TF_augment import TFAug

iter = 20
#
# dali_augment = DALIAug(Iterations = iter)
# dali_augment.run()
#
# DALI_time = []
# for pipe_img_per_sec in dali_augment.all_img_per_sec:
#     DALI_time.append(sum(pipe_img_per_sec) / len(pipe_img_per_sec))
#     print("Total average %7.1f img/s" % (sum(pipe_img_per_sec) / len(pipe_img_per_sec)))

tf_augment =TFAug(Iterations = iter)
tf_augment.run()

all_time = [tf_augment.all_tf_flip_time, tf_augment.all_tf_brightness_time,
            tf_augment.all_tf_contrast_time, tf_augment.all_tf_saturation_time]
tf_time = []
for op_time in all_time:
    tf_time.append(sum(op_time) / len(op_time))
    print("Total average %7.1f img/s" % (sum(op_time) / len(op_time)))

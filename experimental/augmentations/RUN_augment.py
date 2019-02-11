from DALI_augment import DALIAug
from TF_augment import TFAug
from OpenCV_augment import OpenCVAug
from ImgAug_augment import ImgAugAug

flip = True
bri = True
con = True
sat = True

OpenCV = False
DALI = False
TF = False
IA = True

DALI_image_dir = "/home/vzinoviev/Downloads/test_images"
image_dir = "/home/vzinoviev/Downloads/test_images/val2017/*.jpg"
iter = 20
resize_dim = (256, 256)

cv_augment = None
dali_augment = None
tf_augment = None
ia_augment = None

# Running tests

if OpenCV:
    cv_augment = OpenCVAug(Iterations=iter, image_dir=image_dir, dim=resize_dim)
    cv_augment.run(flip=flip, bri=bri, con=con, sat=sat)

if DALI:
    dali_augment = DALIAug(Iterations=iter, image_dir=DALI_image_dir, dim=resize_dim)
    dali_augment.run(flip=flip, bri=bri, con=con, sat=sat)

if TF:
    tf_augment = TFAug(Iterations=iter, image_dir=image_dir, dim=resize_dim)
    tf_augment.run(flip=flip, bri=bri, con=con, sat=sat)

if IA:
    ia_augment = ImgAugAug(Iterations=iter, image_dir=image_dir, dim=resize_dim)
    ia_augment.run(flip=flip, bri=bri, con=con, sat=sat)

# Printing results

if OpenCV:
    print("\nOpenCV")

    for pipe_img_per_sec in cv_augment.all_img_per_sec:
        print("%s: %7.1f img/s" % (pipe_img_per_sec[0], pipe_img_per_sec[1]))

if DALI:
    print("\nNVIDIA DALI")

    for pipe_img_per_sec in dali_augment.all_img_per_sec:
        print("%s: %7.1f img/s" % (pipe_img_per_sec[0], pipe_img_per_sec[1]))

if TF:
    print("\nTensorFlow")

    for pipe_img_per_sec in tf_augment.all_img_per_sec:
        print("%s: %7.1f img/s" % (pipe_img_per_sec[0], pipe_img_per_sec[1]))

if IA:
    print("\nImgAug")

    for pipe_img_per_sec in ia_augment.all_img_per_sec:
        print("%s: %7.1f img/s" % (pipe_img_per_sec[0], pipe_img_per_sec[1]))

print("\nDALI boost: ")
for i in range(len(dali_augment.all_img_per_sec)):
    if OpenCV:
        print("%s: %7.1f" % (cv_augment.all_img_per_sec[i][0],
                                   dali_augment.all_img_per_sec[i][1]/cv_augment.all_img_per_sec[i][1]))
    if TF:
        print("%s: %7.1f" % (tf_augment.all_img_per_sec[i][0],
                                   dali_augment.all_img_per_sec[i][1]/tf_augment.all_img_per_sec[i][1]))
    if IA:
        print("%s: %7.1f" % (ia_augment.all_img_per_sec[i][0],
                                   dali_augment.all_img_per_sec[i][1]/ia_augment.all_img_per_sec[i][1]))

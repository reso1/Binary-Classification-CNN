"""
Utils and configurations file
"""
import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob

#---------------------------------------- configurations ----------------------------------------#
""" Dataset """
tf.app.flags.DEFINE_string('FILE_WRITER_DIR', '../model/tensorboard/',
                           """Directory of file writer """)
                           
tf.app.flags.DEFINE_string('CHECKPOINT_DIR', '../model/checkpoints/',
                           """Directory of checkpoint """)

tf.app.flags.DEFINE_string('TRAIN_DATA_DESCRIPTION_FILE', '../dataset/train.txt',
                           """Base directory of dataset """)

tf.app.flags.DEFINE_string('VALID_DATA_DESCRIPTION_FILE', '../dataset/val.txt',
                           """Directory of file writer """)

tf.app.flags.DEFINE_string('QUAL_IMAGE_FILE', '../dataset/qualified-downsampled/*.jpg',
                           """Directory of file writer """)
                           
tf.app.flags.DEFINE_string('UNQU_IMAGE_FILE', '../dataset/unqualified-downsampled/*.jpg',
                           """Directory of checkpoint """)

tf.app.flags.DEFINE_integer('NUM_CLASSES', 2,
                            """Numer of classes to classify.""")

""" Training parameters """
tf.app.flags.DEFINE_integer('NUM_EPOCHS', 100,
                            """Number of epochs to run.""")

tf.app.flags.DEFINE_integer('BATCH_SIZE', 10,
                            """Batch size.""")

tf.app.flags.DEFINE_float('LEARNING_RATE', 1e-6,
                            """Batch size.""")

tf.app.flags.DEFINE_float('DROPOUT_RATE', 0.5,
                            """Dropout Layer drop rate.""")

tf.app.flags.DEFINE_integer('DISPLAY_STEP', 20,
                            """Display frequency between steps.""")

#---------------------------------------- I/O functions ----------------------------------------#
def generate_image_label_train_validate_file():
    """ Generate train&validate image&label description file from file directory """
    FLAGS = tf.app.flags.FLAGS
    # create train data descrition file
    with open(FLAGS.TRAIN_DATA_DESCRIPTION_FILE, 'a+') as f:
        f.truncate()
        for fn in glob(FLAGS.QUAL_IMAGE_FILE):
            if fn.find('0') == -1:
                f.write(fn + ' 0\n')
        for fn in glob(FLAGS.UNQU_IMAGE_FILE):
            if fn.find('0') == -1:
                f.write(fn + ' 1\n')

    # create validation data descrition file
    with open(FLAGS.VALID_DATA_DESCRIPTION_FILE, 'a+') as f:
        f.truncate()
        for fn in glob(FLAGS.QUAL_IMAGE_FILE):
            if fn.find('0') != -1:
                f.write(fn + ' 0\n')
        for fn in glob(FLAGS.UNQU_IMAGE_FILE):
            if fn.find('0') != -1:
                f.write(fn + ' 1\n')

    # create test data descrition file
    with open(FLAGS.OUTPUT_DATA_DESCRIPTION_FILE, 'a+') as f:
        f.truncate()
        for fn in glob('/home/tortes/pycharm_program/Image/data/processed/testdata/*.jpg'):
            f.write(fn + ' 0\n')

def remove_dir(path):
    fn = os.listdir(path)
    if fn is None or fn == []:
        return

    for i in fn:
        os.remove(path + i)

def read_raw_image(path):
    """read raw images and get normalized units data"""
    # read image and divided into units.
    units = raw_images_2_units(path)
    # do rotate, template matching and downsample, finally we got a list of 256x64 units.
    return rectify(units)

def read_raw_image_single(path):
    units = []
    img = cv2.imread(path, 0)  # read in gray scale
    img = img[500:,:]               # cut off top area
    # divide into 3 units
    for i in range(3):
        dst = img[:, 500*(i+1):500*(i+2)]
        units.append(augment(dst))

    return downsample(rectify(units))

def write_output():
    pass

#---------------------------------------- opencv image functions ----------------------------------------#
def raw_images_2_units(path):
    """read and simply divide the original image to 3 units"""
    imgs = []
    for fn in os.listdir(path):
        img = cv2.imread(path+fn, 0)  # read in gray scale
        img = img[500:,:]               # cut off top area
        # divide into 3 units
        for i in range(3):
            dst = img[:, 500*(i+1):500*(i+2)]
            imgs.append(augment(dst))
    return imgs

def augment(img): 
    """do image augmentation""" 
    # laplacian kernel
    kernel = np.array(
        [[0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]]
    )

    # Global Histogram Equalize (GHE)
    cv2.equalizeHist(img, img)
    return cv2.filter2D(img, 8, kernel)

def hough(img, threshold=200):
    """hough lines detect"""
    ret = []
    for i in img:
        edges = cv2.Canny(i, 100, 200)                   # edge detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold, max_theta=np.pi/6)    # hough lines
        if lines is not None and lines != []:
            for line in lines:
                ret.append(line[0])
    return ret

def rotate(img, w_rho=0.5, w_theta=0.5): 
    """do image rotate"""
    ret = []
    rank = []
    lines = hough(img, 200)
    v_lines = list(filter(lambda l: l[0]>100, lines))
    if v_lines == []:
        return img
    even_rho = even_theta = 0
    for line in v_lines:
        even_rho += line[0]
        even_theta += line[1]
    even_rho /= len(v_lines)
    even_theta /= len(v_lines)
    for idx, line in enumerate(v_lines):
        rank.append([w_rho*abs(line[0]/even_rho) - w_theta*abs(line[1]-even_theta)/even_theta, idx])
    rank.sort(reverse=True)
    for i in img:
        rows,cols = i.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), np.tan(10), 1)
        ret.append(cv2.warpAffine(i, M, (cols,rows)))
    return ret

def template_match(img, template):
    """do template match"""
    img2 = img.copy()
    w, h = template.shape[::-1]
    img = img2.copy()
    # Apply template Matching
    res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    return img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

def rectify(img):
    """do image rotation rectify """
    ret = []
    count = 1
    for idx in range(int(len(img)/3)):
        rot = rotate(img[3*idx:3*(idx+1)], 0.5, 0.5)
        for i in range(3):
            dst = template_match(rot[i], cv2.imread('../dataset/template.jpg', 0))
            ret.append(dst[150:1174, 0:256])
            count += 1 
    return ret

def downsample(img):
    """downsample to 256x64"""
    ret = []
    for i in img:
        ret.append(cv2.pyrDown(cv2.pyrDown(i)))
    return ret

def binarize(img):
    """do image binarize"""
    ret = []
    for i in img:
        threshold_imgs = []
        # adaptive binary threshold
        threshold_imgs.append(
            cv2.adaptiveThreshold(i,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,5)
        ) 
        ret.append(threshold_imgs)
    return ret

def sobel_x(img):
    """do sobel_x conv"""
    return cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

def distort(imgs):      
    """do random brightness, contrast distortion to scale the datasets by 4 times """
    sess=tf.Session()
    ret = []
    l = len(imgs)
    for idx, img in enumerate(imgs):
        i = tf.reshape(img, [256, 64, 1])
        d_brightness = tf.image.random_brightness(i, max_delta=15)
        d_contrast = tf.image.random_contrast(i, lower=0.7, upper=1.3)
        d_brightness_contrast = tf.image.random_contrast(d_brightness, lower=0.7, upper=1.3)

        cv2.imwrite('./dataset/eval/' + str(idx) +'.jpg', i.eval(session=sess))
        cv2.imwrite('./dataset/eval/' + str(idx+l) +'.jpg', d_brightness.eval(session=sess))
        cv2.imwrite('./dataset/eval/' + str(idx+2*l) +'.jpg', d_contrast.eval(session=sess))
        cv2.imwrite('./dataset/eval/' + str(idx+3*l) +'.jpg', d_brightness_contrast.eval(session=sess))

        ret.append(i)
        ret.append(d_brightness)
        ret.append(d_contrast)
        ret.append(d_brightness_contrast)

    return ret
  
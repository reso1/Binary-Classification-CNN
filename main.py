import sys
import src.test
import src.utils as utils

# read images from command line args
image_dir = str(sys.argv[1])
units = utils.read_raw_image(image_dir)


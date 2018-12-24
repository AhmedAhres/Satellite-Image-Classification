import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import re

# Loads an image
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

# Crops an image used when we create patches
def cut_image_for_submission(image, w, h, stride, padding):
    list_patches = []
    width = image.shape[0]
    height = image.shape[1]
    image = np.lib.pad(image, ((padding, padding), (padding, padding), (0,0)), 'reflect')
    for i in range(padding,height+padding,stride):
        for j in range(padding,width+padding,stride):
            image_patch = image[j-padding:j+w+padding, i-padding:i+h+padding, :]
            list_patches.append(image_patch)
    return list_patches

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

# Reads an image and  outputs the label that should go into the submission file
def mask_to_submission_strings(model, filename):
    img_number = int(re.search(r"\d+", filename).group(0))
    image = load_image(filename)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    labels = model.classify(image)
    labels = labels.reshape(-1)
    patch_size = 16
    count = 0
    print("Processing image => " + filename)
    for j in range(0, image.shape[2], patch_size):
        for i in range(0, image.shape[1], patch_size):
            label = int(labels[count])
            count += 1
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

def create_patches(X, patch_size, stride, padding):
    img_patches = np.asarray([cut_image_for_submission(X[i], patch_size, patch_size, stride, padding) for i in range(X.shape[0])])
    # Linearize list
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4])
    return img_patches 

def group_patches(patches, num_images):
    return patches.reshape(num_images, -1)

# Create the csv file
def generate_submission(model, submission_filename, *image_filenames):
    """ Generate a .csv containing the classification of the test set. """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(model, fn))

# Crops an image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

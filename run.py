# Import helpers and the ModelCNN class
from helpers import *
from cnn_model import cnn_model

# Instantiate the model
model = cnn_model(shape = (72,72,3))

# Load the model
model.load('final_model.h5')

# Print a summary to make sure the correct model is used
model.model.summary()

# We add all test images to an array, used later for generating a submission
image_filenames = []
for i in range(1, 51):
    image_filename = 'provided/test_set_images/test_'+ str(i) +'/test_' + str(i) + '.png'
    image_filenames.append(image_filename)

# Set-up submission filename
submission_filename = 'final_submission.csv'

# Generates the submission
generate_submission(model, submission_filename, *image_filenames)

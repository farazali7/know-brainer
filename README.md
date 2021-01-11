[hemorrhage-types]: ./eda_assets/ic-hemorrhage-types.png "Intracranial Hemorrhage Types"
[target-distribution]: ./eda_assets/target-da.png "Target Distribution"
[window-distribution]: ./eda_assets/window-distribution.png "Window Distribution"
[pixel-distribution]: ./eda_assets/pixel-distribution.png "Pixel & HU Units Distribution"
[dicom-windows]: ./eda_assets/dicom-windows.png "DICOM CT Windows"
[brain-px-pct]: ./eda_assets/brain-pixel-pct.png "Brain Pixel Percentages"
[train-result]: ./eda_assets/train-result.png "Train Results"

# Know Brainer
A machine learning model for detecting different types of intracranial hemorrhages.

## Problem Definition
The objective of this model was to leverage deep
learning for detecting the presence of hemorrhage (bleeding) within 
a patient's skull (intracranial) while also identifying the
correct subtype of hemorrhage. The dataset used was provided as part
of the related RSNA Intracranial Hemorrhage challenge, and
consisted of over 600k DICOM files containing patient head CT
scans. The different types of intracranial hemorrhages are:

![hemorrhage-types]

In order to start tackling this problem, the data available
was first analyzed.

## Exploratory Data Analysis
The data analysis began with looking at the distribution
of targets within the dataset.

![target-distribution]

Here are the main takeaways from the above plots:
* The first plot shows there are much more negative instances
in the dataset than positive ones. This might make it
more difficult to train and improve a classifier's 
Recall without a lot of positive data instances.
* The second plot highlights how there are some samples
with multiple hemorrhage subtypes in them.
* The third shows the imbalance across our label sets, 
particularly, we do not have much data for the epidural type.

The information contained in CT scans is usually in reference
to Hounsfield Units (HU) from the Hounsfield scale. 
This scale essentially describes an object's 
radiodensity. For example, air has an HU value of -1000,
while some tissue might have an HU value within 8-70.

![pixel-distribution]

The above plot shows that the raw pixel values in the
DICOM files seem to have a mode around 0. Once these
values are linearly transformed (through two properties
given within the DICOM metadata of a file: 
RescaleIntercept and RescaleSlope), the mode makes more
sense at around -1000 (air). This also shows
that many of the images contain a lot of empty space
alongside the patient's brain. 

Human eyes are only able to detect about a 6% change in
greyscale. Looking at the HU unit distribution above,
it seems some images can have as wide of a HU range as 2000.
Fitting 256 different gray values in that ranges results
in us only detecting a change in the image every 120 ish
HU's. However, changes can happens in smaller ranges 
such as the 8-70 for brain matter. To combat this, 
radiologists use a technique called Windowing. They
define a range of HU's and spread the 256 gray pixel
shades across that smaller range. Determining a good
window range comes from deciding the window level
(middle of range) and window width(range length).

![window-distribution]

While DICOM files contain this value as well, the ones
in the dataset are not uniform (they probably come from
different sources/radiologists). This introduces a source
of variation not intrinsic to the dataset. So, if we 
want to use windowing to generate images for our model
and to help us visualize results, we need to use
standard defined windows for each image.

![dicom-windows]

The last plot combines the brain, subdural, and bone
windows into three channels for an RGB image. This
is the image type that was fed to the model.

![brain-px-pct]

Finally, we can see that there a lot of images that
contain little brain matter (< 0.2) and probably not much
useful information for the model. 

## Image Preprocessing + Data Cleaning
These are some approaches that were taken to help clean
the dataset a bit further and make it more plausible
to train given limited computation resources.
* DICOM files with incorrect RescaleIntercept values
were fixed
* Images with little brain pixel percentages were
removed
* Images were cropped to mostly brain area via the 
following algorithm implemented in OpenCV
    1. Get binary image by converting input image to
    grayscale and using Otsu's thresholding
    2. Create an elliptical shaped kernel and morph
    close to fill in contours.
    3. Get contours and filter them using contour
    approximation and area, then get minimum enclosing
    circle to get a perfect circle.
    4. Draw the circle onto the a blank mask.
    5. Get bounding box of ROI (circle) on blank mask
    then crop to these coordinates.
    6. Bitwise-and on mask and image to get result.
* Oversample the epidural subtype to help model
identify it better

## Model Structure & Performance
The model was based primarily on the pre-trained
ResNet50 model. A separate head was created and put on
top of the ResNet50 model. For training, the dataset
was split into train-validation sets based on multi-label
stratified shuffle split. The model's base was
frozen and the head was trained for a single epoch
on the entire dataset, and then fine-tuned by 
unfreezing the base, and training the entire model 
(including the ResNet layers) on a subset of the dataset.
Training results below.

![train-result]

## Future Work
* Train model on full dataset entirely
* Instead of windowing input, leverage CNN's property of
receiving floating point input data, and giving the
entire set of HU values
    * Might require another form of scaling
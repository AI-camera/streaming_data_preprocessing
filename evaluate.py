import os
from utils import *
from camera import Camera
import improved_dcp
import cv2

# from zero_dce_lle import ZeroDCELLE

DEFAULT_LABELS = 'cfg/coco91.names'
labels = get_classes(DEFAULT_LABELS)
# Good car image: 20152409
# annotation_dir = "./annotation/ExDark_Annno/Car"
annotation_dir = "./annotation/ExDark_Annno/People"
# annotation_dir = "./annotation/car_subset"
# image_dir = "./images/BC_FBS/BC_FBS/Car/"
# image_dir = "./images/BC_CR_RGB/BC_CR_RGB/Car/"
# image_dir = "./images/Car/"
image_dir = "./images/People"
selected_classes = ["people"]

# different values for iou threshold
pred_threshold = []
for i in range(0,20):
    i = i*0.25
    pred_threshold.append(1-i)

print(pred_threshold)
iou_threshold = 0.5
annotation_dict = {}

true_pos = [0 for x in range(len(pred_threshold))]
false_pos = [0 for x in range(len(pred_threshold))]
false_neg = [0 for x in range(len(pred_threshold))]

# Annotation uses XYWH format
# Extract annotation from annotation_dir
for filename in sorted(os.listdir(annotation_dir)):
    # Open the files in the annotation directory
    with open(os.path.join(annotation_dir,filename)) as f:
        # Read all lines in text file
        lines = f.readlines()
        # Remove the first line used for Annotation tool data
        lines = lines[1:]
        # Strip the '.txt' at the end of annotation filename to get image filename
        image_filename = filename[:-4]
        
        # Filename customization (They differs a bit from annotation)
        # image_filename = image_filename.split(".")
        # image_filename = image_filename[0] + "_enhanced.png"
        
        # Create empty object_list
        object_list = []
        # Loop every line in lines
        for line in lines:
            # Split the line using space
            line = line.split()
            # Get name of object
            object_name = line[0].lower()
            if object_name not in selected_classes:
                continue
            # Get bounding box of object (XYWH format)
            bbX = int(line[1])
            bbY = int(line[2])
            bbW = int(line[3])
            bbH = int(line[4])
            # convert the XYWH format into X1Y1X2Y2 format
            bbW, bbH = [bbX+bbW, bbY + bbH]
            # Put the data into a bounding box
            bbox = [object_name, bbX, bbY, bbW, bbH]
            # Add the bounding box into object_list
            object_list.append(bbox)
        
        # Add the object_list to annotation_dict, with key being image_filename
        annotation_dict[image_filename] = object_list.copy()
        f.close()

# Create a camera, but only use it's MobileDet detector
# don't call camera.run()
fps = 24
fps_max = 24
camera = Camera(fps_max,"./sample/home_night_01.mp4")
camera.set_selected_classes(selected_classes)
camera.set_frame_skip = 0
camera.set_detect_box(0,0,1,1)
# camera.set_preprocess_functions([camera.lowlight_enhance])
camera.lle_scale = 1
camera.lle_pyramid = True

# Use keys in annotation_dict as filename
for image_filename in annotation_dict.keys():
    print("Processing %s" % (image_filename))
    # Read the file 
    image = cv2.imread(os.path.join(image_dir,image_filename))
    # Use camera detector to detect object on image
    # image = improved_dcp.lowlight_enhance_modified(image,pyramid=True)
    # image = improved_dcp.lowlight_enhance(image,pyramid=True)
    pred_boxes, scores, pred_classes = camera.detector.detect(image, camera.detect_box, selected_classes = camera.selected_classes)
    # num_match increase by 1 every match between pred_boxes and annotated boxes
    num_match = [0 for x in range(len(true_pos))]
    for i in range(len(pred_boxes)):
        # If the confidence score is less than threshold, continue
        # if scores[i] < pred_threshold:
        #     continue
        # If a box does not find a match, plus 1 for false pos
        for bbox in annotation_dict[image_filename]:
            # Check if the object name in bbox[0] is the same
            # If not, continue
            if bbox[0] != labels[pred_classes[i]]:
                continue
            # Compute the iou of predicted box and annotated box
            # if it reach threshold, plus 1 for true_pos
            # Perform this for all iou/pred threshold values
            for j in range(len(pred_threshold)):
                # if score > threshold, determine it's true/false_pos aspect
                if (scores[i]) > pred_threshold[j]:
                    # If a 
                    if iou(bbox[1:],pred_boxes[i]) >= iou_threshold:
                        true_pos[j] += 1
                        num_match[j] += 1
                    # If a box does not find a match, plus 1 for false pos
                    else:
                        false_pos[j] += 1

                        
    # Increase number of false_neg by number_of_annotated_box - number_of_match
    # There is *negligible* amount of dupplicate match since we did nms on pred_boxes
    for j in range(len(true_pos)):
        false_neg[j] += len(annotation_dict[image_filename]) - num_match[j]

print("true_pos = ")
print(true_pos)
print("false_pos = ")
print(false_pos)
print("false_neg = ")
print(false_neg)
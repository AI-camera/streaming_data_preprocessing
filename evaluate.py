import os
from utils import *
from camera import Camera
import cv2

DEFAULT_LABELS = 'cfg/coco91.names'
labels = get_classes(DEFAULT_LABELS)
# Good car image: 20152409
# annotation_dir = "./annotation/ExDark_Annno/Car"
annotation_dir = "./annotation/ExDark_Annno/People"
# annotation_dir = "./annotation/car_subset"
image_dir = "./images/People"
selected_classes = ["people"]

true_pos = 0
false_pos = 0
false_neg = 0

iou_threshold = 0.35
pred_threshold = 0.6
annotation_dict = {}
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

# Create a camera
fps = 24
fps_max = 24
camera = Camera(fps_max,"./sample/home_night_01.mp4")
camera.set_selected_classes(selected_classes)
# camera.set_preprocess_functions([camera.lowlight_enhance])

# Use keys in annotation_dict as filename
for image_filename in annotation_dict.keys():
    print("Processing %s" % (image_filename))
    # Read the file 
    image = cv2.imread(os.path.join(image_dir,image_filename))
    pred_boxes, scores, pred_classes = camera.detector.detect(image, camera.detect_box, selected_classes = camera.selected_classes)
    # num_match increase by 1 every match between pred_boxes and annotated boxes
    num_match = 0
    for i in range(len(pred_boxes)):
        # If the confidence score is less than threshold, continue
        if scores[i] < pred_threshold:
            continue
        # If a box does not find a match, plus 1 for false pos
        match_found = False
        for bbox in annotation_dict[image_filename]:
            # Check if the object name in bbox[0] is the same
            # If not, continue
            if bbox[0] != labels[pred_classes[i]]:
                continue
            # Compute the iou of predicted box and annotated box
            # if it reach threshold, plus 1 for true_pos
            if iou(bbox[1:],pred_boxes[i]) > iou_threshold:
                true_pos += 1
                match_found = True
        
        if not match_found:
            false_pos += 1
        else:
            num_match += 1
        
    # Increase number of false_neg by number_of_annotated_box - number_of_match
    # There is *negligible* amount of dupplicate match since we did nms on pred_boxes
    false_neg += len(annotation_dict[image_filename]) - num_match


print(true_pos)
print(false_pos)
print(false_neg)
precision = true_pos/(true_pos+false_pos)
recall = true_pos/(true_pos+false_neg)
f1_score = (2*precision*recall)/(precision+recall)
print("Precision: %.2f" % (precision))
print("Recall: %.2f" % (recall))
print("F1 score: %.2f" % (f1_score))
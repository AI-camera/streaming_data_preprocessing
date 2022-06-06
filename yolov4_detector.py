import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DISPLAY'] = ':0'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags
# from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect(interpreter, input_tensor):
    """Run detection on an input image.
    Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.
    Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # We use the original model for pre-processing, since the TFLite model doesn't
    # include pre-processing.
    # preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    input_tensor = tf.expand_dims(tf.convert_to_tensor(input_tensor, dtype=tf.float32), axis=0)
    input_tensor = np.resize(input_tensor,(1,320,320,3))
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])
    scores = interpreter.get_tensor(output_details[0]['index'])
    return boxes, classes, scores

def main(_argv):
    # Definition of the parameters
    nms_max_overlap = 1.0

    frame = cv2.imread("images/car_image_01.png")

    saved_model_loaded = tf.saved_model.load(DEFAULT_WEIGHT_PATH, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    #Runs on tensorflow only
    batch_data = tf.constant(frame)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)

    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]

    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    allowed_classes = ['car']

    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

    #initialize color map
    cmap = plt.get_cmap('tab20b')

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = utils.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]       


    # calculate frames per second of running detections
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)
    result = np.asarray(frame)
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
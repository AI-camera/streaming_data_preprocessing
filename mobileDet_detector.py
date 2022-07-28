# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to detect objects in a given image.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh detect_image.py

python3 examples/detect_image.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels test_data/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output ${HOME}/grace_hopper_processed.bmp
```
"""
from PIL import Image
from utils import *
import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

from pycoral import common
from pycoral import detect
# from pycoral.edgetpu import make_interpreter

DEFAULT_LABELS = 'cfg/coco91.names'
DEFAULT_MODEL = 'models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'
EDGETPU_SHARED_LIB = "libedgetpu.so.1"

class Detector:
  def __init__(self, quantization=True, threshold=0.6):
    self.labels = get_classes(DEFAULT_LABELS)
    self.model_path = DEFAULT_MODEL
    self.edge_tpu = True
    self.interpreter = self.make_interpreter()
    self.interpreter.allocate_tensors()
    self.colors = np.random.uniform(30, 255, size=(len(self.labels), 3))
    self.threshold = threshold
    self.preprocess_functions = []
  
  def make_interpreter(self):
    ''' 
    Create an interpreter 
    - Args:
        * model_path: path to model file (must be tflite)
        * edge_tpu: enable edge tpu
    ''' 
    # Load the TF-Lite model and delegate to Edge TPU
    if self.edge_tpu:
        interpreter = tflite.Interpreter(model_path=self.model_path,
                experimental_delegates=[
                    tflite.load_delegate(EDGETPU_SHARED_LIB)
                    ])
    else:
        interpreter = tflite.Interpreter(model_path=self.model_path)

    return interpreter
  
  def set_preprocess_functions(self,preprocess_functions):
    '''Set a LIST of preprocess_functions'''
    self.preprocess_functions = preprocess_functions

  def detect(self,image,crop_box=None, selected_classes = ["car","motorbike"]):
    '''
      - Detect objects in image
      - Params:
        * image: ndarray image
      - Returns:
        * bboxes, scores, classes
    '''
    x1,y1 = denormalize_coordinate(image,crop_box[0])
    x2,y2 = denormalize_coordinate(image,crop_box[1])
    image = imcrop(image,x1,y1,x2,y2)
      
    _, scale = common.set_resized_input(self.interpreter, (image.shape[1],image.shape[0]), lambda size: cv2.resize(image,size,interpolation = cv2.INTER_AREA),self.preprocess_functions)

    self.interpreter.invoke()
    objs = detect.get_objects(self.interpreter, self.threshold, scale)

    bboxes = []
    classes = []
    scores = []

    for obj in objs:
      if (self.labels[obj.id] in selected_classes) or len(selected_classes) == 0:
        bboxes.append(((obj.bbox.xmin,obj.bbox.ymin),(obj.bbox.xmax,obj.bbox.ymax)))
        classes.append(obj.id)
        scores.append(obj.score)
    
    if len(bboxes) > 0:
        bboxes, scores, classes = nms_boxes(bboxes, scores, classes)
    
    return bboxes, scores, classes
 
  def draw_boxes(self, image, boxes, scores, classes,offset_x=0,offset_y=0):
    '''
    - Draw predicted bounding boxes on image
    - Args:
        * image: input image
        * boxes: bounding boxes
        * scores: scores of bounding boxes
        * classes: classes of bounding boxes
        * class_names: set of class names
        * offset_x: base x coordinate of cropbox
        * offset_y: base y corrdinate of cropbox
    '''
    i = 0
    for topleft, botright in boxes:
        # Detected class
        cl = int(classes[i])
        # This stupid thing below is needed for opencv to use as a color
        color = tuple(map(int, self.colors[cl])) 

        # Box coordinates
        # Add offset in case only a region of image is fed to model
        topleft = (int(topleft[0]+offset_x), int(topleft[1]+offset_y))
        botright = (int(botright[0]+offset_x), int(botright[1]+offset_y))
        cv2.rectangle(image, topleft, botright, color, 1)
        textpos = (topleft[0]-2, topleft[1] - 3)
        score = scores[i] * 100
        cl_name = self.labels[cl]

        text = f"{cl_name} ({score:.1f}%)"
        cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,
                0.45, color, 1, cv2.LINE_AA)
        i += 1
    
    text = f"object count: {i}"
    textpos=(0,50)
    cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1, cv2.LINE_AA)

    return image
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

from pycoral_lib.pycoral_utils.adapters import common
from pycoral_lib.pycoral_utils.adapters import detect
from pycoral_lib.pycoral_utils.utils.edgetpu import make_interpreter

DEFAULT_LABELS = 'cfg/coco91.names'
DEFAULT_MODEL = 'models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

class Detector:
  def __init__(self, quantization=True, threshold=0.5):
    self.labels = get_classes(DEFAULT_LABELS)
    self.model = DEFAULT_MODEL
    self.interpreter = make_interpreter(self.model)
    self.interpreter.allocate_tensors()
    self.colors = np.random.uniform(30, 255, size=(len(self.labels), 3))
    self.threshold = threshold

  def detect(self,image,crop_box=None, frame_skip=0, selected_classes = ["car","motorbike"],preprocess_functions = []):
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
    for function in preprocess_functions:
      image = function(image)
    
    image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image.astype('uint8'))
    _, scale = common.set_resized_input(
    self.interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    self.interpreter.invoke()
    objs = detect.get_objects(self.interpreter, self.threshold, scale)

    bboxes = []
    classes = []
    scores = []

    for obj in objs:
      if self.labels[obj.id] in selected_classes:
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
    vehicle_names = ["car","motorbike"]
    vehicle_count = 0
    for topleft, botright in boxes:
        # Detected class
        cl = int(classes[i])
        # This stupid thing below is needed for opencv to use as a color
        color = tuple(map(int, self.colors[cl])) 

        # Box coordinates
        # Add offset in case only a region of image is fed to model
        topleft = (int(topleft[0]+offset_x), int(topleft[1]+offset_y))
        botright = (int(botright[0]+offset_x), int(botright[1]+offset_y))
        cv2.rectangle(image, topleft, botright, color, 2)
        textpos = (topleft[0]-2, topleft[1] - 3)
        score = scores[i] * 100
        cl_name = self.labels[cl]
        
        if cl_name in vehicle_names:
            vehicle_count +=1

        text = f"{cl_name} ({score:.1f}%) (position: {topleft[0]:.1f})"
        cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,
                0.45, color, 1, cv2.LINE_AA)
        i += 1
    
    text = f"vehicle count: {vehicle_count}"
    textpos=(0,50)
    cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1, cv2.LINE_AA)

    return image
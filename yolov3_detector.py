# from tkinter import ANCHOR
# from cv2 import CV_8UC1
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
from time import time
from utils import *
import string
import pytesseract
import dcp_dehaze

EDGETPU_SHARED_LIB = "libedgetpu.so.1"
alphabets = string.digits + string.ascii_lowercase
blank_index = len(alphabets)

DEFAULT_CLASSES_CFG = "./cfg/coco.names"
DEFAULT_ANCHORS_CFG = "./cfg/tiny_yolo_anchors.txt"
DEFAULT_MODEL_PATH = "./models/quant_coco-tiny-v3-relu_edgetpu.tflite"

class Detector:
    def __init__(self,classes_cfg=DEFAULT_CLASSES_CFG,
                anchors_cfg=DEFAULT_ANCHORS_CFG,
                model_path=DEFAULT_MODEL_PATH,
                edge_tpu=True,
                quantization=True,
                threshold=0.3):
        '''
        Create a detector object 
        - Args:
            * classes_cfg: Path to classes config
            * anchors_cfg: Path to anchors config
            * model_path: Path to model file (must be tflite for now)
            * edge_tpu: Set to True to use edge tpu. Default: True
            * quantization: Set to True if model is int8 quantized. Default: True
            * threshold: Inference threshold
        '''
        self.classes = get_classes(classes_cfg)
        self.anchors = get_anchors(anchors_cfg)
        self.model_path = model_path
        self.colors = np.random.uniform(30, 255, size=(len(self.classes), 3))
        self.edge_tpu = edge_tpu
        self.quantization = quantization
        self.threshold = threshold
        self.preprocess_functions = []

        self.interpreter = self.make_interpreter()
        self.interpreter.allocate_tensors()

    def set_preprocess_functions(self,preprocess_functions):
        self.preprocess_functions = preprocess_functions

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

    def tesseract_ocr(image):
        text = ""
        # image.convertTo(image, CV_8UC1)
        image[image<0] = 0
        image[image>255] = 255
        image = image.astype("uint8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Morph open to remove noise and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        invert = 255 - opening

        # Perform text extraction
        cv2.imwrite("tesseract.png",invert)
        text = pytesseract.image_to_string(invert, config='--psm 6')
        return text

    def inference(self, img):
        '''
        Make inference base on an interpreter
        - Args:
            * intepreter: a tflite interpreter
            * img: the input image
            * anchors: set of predefined anchors
            * n_classes: set of classes
            * threshold: inference threshold
        - Returns:
            * boxes, scores, classes
        '''
        input_details, output_details, net_input_shape = self.get_interpreter_details()
        # print("yolov3_detector.py inference() output_details")
        # print(output_details)
        img_orig_shape = img.shape
        # Crop frame to network input shape
        img = letterbox_image(img.copy().astype('uint8'), (416, 416))

        # performs post-reshape preprocessing
        try:
            if self.preprocess_functions is not None:
                for function in self.preprocess_functions:
                    img = function(img)
        except:
            print("yolov3_detector.py Something is wrong with post-reshape preprocessing")

        img = img.astype('uint8')
        # Add batch dimension
        img = np.expand_dims(img, 0)
        
        # For the yolov4 to work
        # img = np.array(img, dtype='int8')

        if not self.quantization:
            # Normalize image from 0 to 1
            img = np.divide(img, 255.).astype(np.float32)

        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], img)

        start = time()

        # Run model
        self.interpreter.invoke()

        inf_time = time() - start
        # print(f"Net forward-pass time: {inf_time*1000} ms.")
        # Retrieve outputs of the network
        out1 = self.interpreter.get_tensor(output_details[0]['index'])
        out2 = self.interpreter.get_tensor(output_details[1]['index'])

        # If this is a quantized model, dequantize the outputs
        if self.quantization:
            # Dequantize output
            o1_scale, o1_zero = output_details[0]['quantization']
            out1 = (out1.astype(np.float32) - o1_zero) * o1_scale
            o2_scale, o2_zero = output_details[1]['quantization']
            out2 = (out2.astype(np.float32) - o2_zero) * o2_scale
        # Get boxes from outputs of network
        start = time()
        _boxes1, _scores1, _classes1 = featuresToBoxes(out1, self.anchors[[3, 4, 5]], 
                len(self.classes), net_input_shape, img_orig_shape, self.threshold)
        _boxes2, _scores2, _classes2 = featuresToBoxes(out2, self.anchors[[1, 2, 3]], 
                len(self.classes), net_input_shape, img_orig_shape, self.threshold)
        inf_time = time() - start
        # print(f"Box computation time: {inf_time*1000} ms.") 

        # This is needed to be able to append nicely when the output layers don't
        # return any boxes
        if _boxes1.shape[0] == 0:
            _boxes1 = np.empty([0, 2, 2])
            _scores1 = np.empty([0,])
            _classes1 = np.empty([0,])
        if _boxes2.shape[0] == 0:
            _boxes2 = np.empty([0, 2, 2])
            _scores2 = np.empty([0,])
            _classes2 = np.empty([0,])
        
        boxes = np.append(_boxes1, _boxes2, axis=0)
        scores = np.append(_scores1, _scores2, axis=0)
        classes = np.append(_classes1, _classes2, axis=0)
        if len(boxes) > 0:
            boxes, scores, classes = nms_boxes(boxes, scores, classes)
        return boxes, scores, classes

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
        # vehicle_names = ["car","motorbike"]
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

            #wpod and ocr
            # if class_names[cl] == "car" or class_names[cl] == "truck":
            #     lpText = ""
            #     car_crop = image[topleft[1]:botright[1]+1,topleft[0]:botright[0]+1]
                
            #     if car_crop is not None and min(car_crop.shape[:2]) is not 0:
            #         # cv2.imwrite("out_images/car_crop_" + str(i) + ".png", car_crop)
            #         # plate_crop = alpr.license_plate_detect(car_crop)  
            #         plate_crop = alpr.license_plate_detect_tflite(car_crop)  
                    
            #     if plate_crop is not None and plate_crop.shape[1] is not 0 and plate_crop.shape[0] is not 0:
            #         # cv2.imwrite("out_images/plate_crop_" + str(i) + ".png", plate_crop)
            #         # lpText = alpr.license_plate_ocr(plate_crop)
            #         lpText = tesseract_ocr(plate_crop)
            #         # print(lpText)
            #     #     if lpText is None:
            #     #         lpText = ""

            # Draw box and class
            cv2.rectangle(image, topleft, botright, color, 1)
            textpos = (topleft[0]-2, topleft[1] - 3)
            score = scores[i] * 100
            cl_name = self.classes[cl]
            
            # if cl_name in vehicle_names:
            #     vehicle_count +=1

            text = f"{cl_name} ({score:.1f}%)"
            cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,
                    0.45, color, 1, cv2.LINE_AA)
            i += 1
        
        text = f"vehicle count: {vehicle_count}"
        textpos=(0,50)
        cv2.putText(image, text, textpos, cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1, cv2.LINE_AA)

        return image

    def get_interpreter_details(self):
        # Get input and output tensor details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_shape = input_details[0]["shape"]

        return input_details, output_details, input_shape

    def detect(self,img,crop_box=None, selected_classes = ["car","motorbike"]):
        '''
        Make inference on an image
        - Args:
            * img: Input image
            * crop_box: Specify a region on image to do inference. Format ((x1,y1),(x2,y2)). Value range: 0-100

            * selected_classes: Classes that are to be detected
        - Return:
            * (boxes, scores, pred_classes)
        '''
        # print("yolov3_detector.py image_inf() ")

        n_classes = len(self.classes)
        img = img.copy()
        
        #Crop the image using (x1,y1) and (x2,y2)
        x1,y1 = denormalize_coordinate(img,crop_box[0])
        x2,y2 = denormalize_coordinate(img,crop_box[1])
        cropped_img = imcrop(img,x1,y1,x2,y2)
        boxes, scores, pred_classes = self.inference(cropped_img)
        # Get the boxes of selected class
        result_list = [(box, score, pred_class) for (box, score, pred_class) in zip(boxes, scores,pred_classes) if self.classes[int(pred_class)] in selected_classes]

        if(len(result_list)>0):
            boxes = [row[0] for row in result_list]
            scores = [row[1] for row in result_list]
            pred_classes = [row[2] for row in result_list]
        else:
            boxes = []
            scores = []
            pred_classes = []
        return boxes, scores, pred_classes
    
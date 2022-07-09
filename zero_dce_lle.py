import numpy as np
import tflite_runtime.interpreter as tflite
from time import time
from utils import *
import cv2 
import tensorflow as tf
import dcp_dehaze

EDGETPU_SHARED_LIB = "libedgetpu.so.1"
MODEL_PATH = "./models/zero_dce_quant_edgetpu.tflite"

class ZeroDCELLE:
    def __init__(self):
        self.interpreter = self.make_interpreter(MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details, self.output_details, self.net_input_shape = self.get_interpreter_details()
        self.session=tf.compat.v1.Session()

    def get_interpreter_details(self):
        # Get input and output tensor details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_shape = input_details[0]["shape"]

        return input_details, output_details, input_shape

    def make_interpreter(self, model_path):
        ''' 
        Create an interpreter 
        - Args:
            * model_path: path to model file (must be tflite)
            * edge_tpu: enable edge tpu
        ''' 
        # Load the TF-Lite model and delegate to Edge TPU
        interpreter = tflite.Interpreter(model_path=model_path,
                experimental_delegates=[
                    tflite.load_delegate(EDGETPU_SHARED_LIB)
                    ])
        return interpreter

    def lowlight_enhance(self,frame):
        # Make the inference
        frame = letterbox_image(frame.copy().astype('uint8'), (512, 512))
        frame = np.expand_dims(frame, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()
        out1 = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Dequantize output
        o1_scale, o1_zero = self.output_details[0]['quantization']
        
        A = (out1.astype(np.float32) - o1_zero) * o1_scale
        frame = frame/255
        r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
        x = frame + r1 * (np.power(frame,2)-frame)
        x = x + r2*(np.power(x,2)-x)
        x = x + r3*(np.power(x,2)-x)
        x = x + r4*(np.power(x,2)-x)
        x = x + r5*(np.power(x,2)-x)		
        x = x + r6*(np.power(x,2)-x)	
        x = x + r7*(np.power(x,2)-x)
        enhance_image = x + r8*(np.power(x,2)-x)
        
        enhance_image = enhance_image*255
        enhance_image[enhance_image>255] = 255
        enhance_image[enhance_image<0] = 0
        enhance_image = np.squeeze(enhance_image)
        return enhance_image.astype('uint8')
        

if __name__ == "__main__":
    zero_dce_lle = ZeroDCELLE()
    frame = cv2.imread("images/Car/2015_02420.jpg")
    start = time()
    output = zero_dce_lle.lowlight_enhance(frame)
    print(f"Inference time zero_dce: {(time() - start):.2f}")
    cv2.imwrite("output/out_dce.png",output)

    start = time()
    output = dcp_dehaze.lowlight_enhance(frame)
    print(f"Inference time dcp: {(time() - start):.2f}")
    cv2.imwrite("output/out_dcp.png",output)
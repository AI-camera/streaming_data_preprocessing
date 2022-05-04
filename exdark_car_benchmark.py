from camera import Camera
from mobileDet_detector import Detector as MobileDetDetector
import os
import cv2

if __name__ == '__main__':
    fps = 24
    fps_max = 24
    camera = Camera(fps_max,"./sample/akihabara_03.mp4",True, detector=MobileDetDetector())
    camera.set_frame_skip(0)

    input_directory = "images/BC_CR_RGB/Car"
    output_directory = "output/Car_BC_CR_RGB"

    for filename in os.listdir(input_directory):
        input_filename = os.path.join(input_directory,filename)
        input_image = cv2.imread(input_filename)
        if input_image is None:
            continue
        print(f"Processing file: {filename}")
        # input_image = dehaze.lowlight_enhance(input_image)
        output = camera.detect_object(input_image)
        
        output_filename = os.path.join(output_directory,filename)
        cv2.imwrite(output_filename,output)

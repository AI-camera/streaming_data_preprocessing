from camera import Camera
from mobileDet_detector import Detector as MobileDetDetector
import os
import cv2
import utils

input_directory = "images/Car_fbs_02/"
output_directory = "output/Car_fbs_02_mobileDet/"
reference_directory = "images/NIND_low_ISO"

fps = 24
fps_max = 24
camera = Camera(fps_max,"./sample/akihabara_03.mp4",True, detector=MobileDetDetector())
camera.set_frame_skip(0)

def iso_denoise_benchmark():
    input_filepaths = []
    reference_filepaths = []

    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
    for filename in sorted(os.listdir(reference_directory)):
        reference_filepaths.append(os.path.join(reference_directory,filename))
    
    i = 0
    for i in range(len(input_filepaths)):
        input_image = cv2.imread(input_filepaths[i])
        reference_image = cv2.imread(reference_filepaths[i])
        if input_image is None:
            continue
        print(f"Processing file: {filename}")
        # input_image = dehaze.lowlight_enhance(input_image)
        output_image = camera.denoise(input_image)

        output_filename = os.path.join(output_directory,filename)
        cv2.imwrite(output_filename,output_image)
        
        psnr = utils.calculate_psnr(reference_image,input_image)
        print(f"PSNR of pair {filename} is {psnr}")

        denoised_psnr = utils.calculate_psnr(reference_image,output_image)
        print(f"Denoised PSNR of pair {filename} is {denoised_psnr}")

        ssim = utils.calculate_ssim(reference_image,input_image)
        print(f"SSIM of pair {filename} is {ssim}")

        denoised_ssim = utils.calculate_ssim(reference_image,output_image)
        print(f"Denoised SSIM of pair {filename} is {denoised_ssim}")

def file_benchmark():
    camera.set_selected_classes(["car"])
    for filename in sorted(os.listdir(input_directory)):
        input_image = cv2.imread(os.path.join(input_directory,filename))
        if input_image is None:
            continue
        print(f"Processing file: {filename}")
        # input_image = dehaze.lowlight_enhance(input_image)
        output = camera.detect_object(input_image)
        
        output_filename = os.path.join(output_directory,filename)
        cv2.imwrite(output_filename,output)

def camera_benchmark():
    frame = camera.get_raw_frame()
    frame = camera.sharpen(frame)

    while True:
        frame = camera.get_raw_frame()
        if frame is None:
            continue           

        i +=1
        if i > 2000:
            break

        if len(inference_time_history) > 10:
            inference_time_history = inference_time_history[1:]

        if i%100 == 0 and i > 0:
            print("camera.py executing " + str(i))
            print("FPS: ", str(1/(sum(inference_time_history)/len(inference_time_history))))

        camera.write_frame_to_output_file(frame)

    camera.out.release()

if __name__ == '__main__':
    file_benchmark()
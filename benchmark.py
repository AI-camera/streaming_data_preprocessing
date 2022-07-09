from camera import Camera
from mobileDet_detector import Detector as MobileDetDetector
import os
import cv2
import utils

# input_directory = "images/eval15/low"
# reference_directory = "images/eval15/high"
# output_directory = "output/NIND_low_ISO/"

input_directory = "images/NIND_high_ISO/"
reference_directory = "images/NIND_low_ISO/"
# output_directory = "output/NIND_low_ISO/"

fps = 24
fps_max = 24
camera = Camera(fps_max,"./sample/akihabara_03.mp4",True, detector=MobileDetDetector())
camera.set_frame_skip(0)

def psnr_ssim_benchmark():
    input_filepaths = []
    reference_filepaths = []

    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
    for filename in sorted(os.listdir(reference_directory)):
        reference_filepaths.append(os.path.join(reference_directory,filename))
    
    i = 0
    with open("./result_lle.txt","w") as f:
        for i in range(len(input_filepaths)):
            print("Processing image %s" % input_filepaths[i])
            input_image = cv2.imread(input_filepaths[i])
            reference_image = cv2.imread(reference_filepaths[i])
            if input_image is None:
                continue
            f.write(f"Processing file: {filename} \n")
            output_image = camera.lowlight_enhance(input_image)
            # output_image = camera.denoise(input_image)

            # output_filename = os.path.join(output_directory,filename)
            # cv2.imwrite(output_filename,output_image)
            
            psnr = utils.calculate_psnr(reference_image,input_image)
            f.write(f"PSNR of pair {filename} is {psnr} \n")

            denoised_psnr = utils.calculate_psnr(reference_image,output_image)
            f.write(f"Processed PSNR of pair {filename} is {denoised_psnr} \n")

            ssim = utils.calculate_ssim(reference_image,input_image)
            f.write(f"SSIM of pair {filename} is {ssim} \n")

            denoised_ssim = utils.calculate_ssim(reference_image,output_image)
            f.write(f"Processed SSIM of pair {filename} is {denoised_ssim} \n")
    f.close()

def entropy_benchmark():
    input_filepaths = []
    reference_filepaths = []

    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
    
    i = 0
    with open("./result_lle.txt","w") as f:
        for i in range(len(input_filepaths)):
            print("Processing image %s" % input_filepaths[i])
            input_image = cv2.imread(input_filepaths[i])
            if input_image is None:
                continue
            f.write(f"Processing file: {filename} \n")

            output_image = camera.lowlight_enhance(input_image)
            
            input_entropy = utils.entropy(input_image)
            output_entropy = utils.entropy(output_image)
            f.write(f"Entropy of {filename} is {input_entropy} \n")
            f.write(f"Processed Entropy of {filename} is {output_entropy} \n")

    f.close()

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
    psnr_ssim_benchmark()
from camera import Camera
from mobileDet_detector import Detector as MobileDetDetector
import os
import cv2
import utils
import dcp_dehaze
import improved_dcp
import time
# input_directory = "output/zero_dce_quantized_lle/"
# output_directory = "output/zero_dce_quantized_lle/"
# reference_directory = "output/zero_dce_lle/high"

input_directory = "images/eval15/low"
reference_directory = "images/eval15/high"
# output_directory = "output/NIND_low_ISO/"

# input_directory = "images/landmark_01/"
# reference_directory = "images/NIND_low_ISO/"
# output_directory = "output/landmark_01/"

# input_directory = "images/Car"
# input_directory = "images/defog_super_resolution/Images"
# output_directory = "output/defog_super_resolution/Images"

# input_directory = "images/defog_super_resolution/ImagesDefoged"
# output_directory = "output/defog_super_resolution/ImagesDefoged"

# input_directory = "images/defog_super_resolution/ImagesDefoged_2"
# output_directory = "output/defog_super_resolution/ImagesDefoged_2"

# input_directory = "images/outdoor/defogged"
# output_directory = "output/outdoor/defogged"

fps = 24
fps_max = 24
camera = Camera(fps_max,"./sample/akihabara_03.mp4",True, detector=MobileDetDetector())
camera.set_frame_skip(0)
camera.set_selected_classes([])
input_filepaths = []
reference_filepaths = []
output_filepaths = []

def psnr_ssim_entropy_benchmark():
    total_psnr = 0
    total_ssim = 0
    total_entropy = 0
    total_std = 0
    total_time = 0
    count = 0

    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
    # for filename in sorted(os.listdir(output_directory)):
    #     output_filepaths.append(os.path.join(output_directory,filename))
    for filename in sorted(os.listdir(reference_directory)):
        reference_filepaths.append(os.path.join(reference_directory,filename))
    
    i = 0
    for i in range(len(input_filepaths)):
        print("Processing image %s" % input_filepaths[i])
        input_image = cv2.imread(input_filepaths[i])
        reference_image = cv2.imread(reference_filepaths[i])
        if input_image is None:
            continue
        
        count += 1
        start = time.time()
        # output_image = cv2.medianBlur(input_image,5)
        # output_image = cv2.GaussianBlur(input_image,(5,5),cv2.BORDER_DEFAULT)
        output_image = improved_dcp.lowlight_enhance(input_image,pyramid=True)
        # output_image = improved_dcp.lowlight_enhance_modified(input_image,pyramid=True)
        total_time += time.time() - start

        # output_image = cv2.imread(output_filepaths[i])
        output_image = output_image.astype("uint8")
        psnr = utils.calculate_psnr(reference_image,output_image)
        total_psnr += psnr

        ssim = utils.calculate_ssim(reference_image,output_image)
        total_ssim += ssim

        entropy = utils.entropy(output_image)
        total_entropy += entropy

        std = utils.standard_deviation(output_image)
        total_std = entropy
    
    print("Average time")
    print(total_time/count)
    print("Average psnr")
    print(total_psnr/count)
    print("Average ssim")
    print(total_ssim/count)
    print("Average entropy")
    print(total_entropy/count)
    print("Average std")
    print(total_std/count)
    
def entropy_benchmark():
    input_filepaths = []
    input_entropy = []
    output_entropy = []

    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
    
    i = 0
    for i in range(len(input_filepaths)):
        print("Processing image %s" % input_filepaths[i])
        input_image = cv2.imread(input_filepaths[i])
        if input_image is None:
            continue

        output_image = camera.lowlight_enhance(input_image)
        
        input_entropy.append(utils.entropy(input_image))
        output_entropy.append(utils.entropy(output_image))

    print("Average input entropy")
    print(sum(input_entropy)/len(input_entropy))
    print("Average output entropy")
    print(sum(output_entropy)/len(output_entropy))

def camera_benchmark():
    frame = camera.get_raw_frame()
    # frame = camera.sharpen(frame)

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

def lowlight_enhance_dir():
    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
        output_filepaths.append(os.path.join(output_directory,filename))

    for i in range(len(input_filepaths)):
        print("Processing image %s" % input_filepaths[i])
        input_image = cv2.imread(input_filepaths[i])
        if input_image is None:
            continue
        output_image = camera.lowlight_enhance(input_image)
        cv2.imwrite(output_filepaths[i],output_image)

def detection_benchmark():
    for filename in sorted(os.listdir(input_directory)):
        input_filepaths.append(os.path.join(input_directory,filename))
        output_filepaths.append(os.path.join(output_directory,filename))

    for i in range(len(input_filepaths)):
        print("Processing image %s" % input_filepaths[i])
        input_image = cv2.imread(input_filepaths[i])
        if input_image is None:
            continue
        output_image = camera.detect_object(input_image)
        cv2.imwrite(output_filepaths[i],output_image)

if __name__ == '__main__':
    psnr_ssim_entropy_benchmark()
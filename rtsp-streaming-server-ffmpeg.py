from camera_manager import CameraManager
import threading
import subprocess as sp
import time

import os
os.environ['DISPLAY'] = ':0'

rtsp_raw = 'rtsp://127.0.0.1:8554/raw_stream'
rtsp_denoised = 'rtsp://127.0.0.1:8554/denoised_stream'
rtsp_lowlight_enhance = 'rtsp://127.0.0.1:8554/lowlight_enhance_stream'
rtsp_motion_adaptive = 'rtsp://127.0.0.1:8554/motion_adaptive_stream'

def raw_stream(camera, id):
    sizeStr = camera.get_sizestr()
    fps = camera.get_fps()
    print(sizeStr)
    rtsp_server = rtsp_raw + "/" + id
    command = ['ffmpeg',
            '-re',
            '-s', sizeStr,
            '-r', str(fps),  # rtsp fps (from input server)
            '-i', '-',
               
            # You can change ffmpeg parameter after this item.
            
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # output fps
            '-g', '50',
            '-c:v', 'libx264',
            '-b:v', '2M',
            '-bufsize', '64M',
            '-maxrate', "4M",
            '-preset', 'veryfast',
            '-rtsp_transport', 'tcp',
            '-segment_times', '5',
            '-f', 'rtsp',
            rtsp_server]

    process = sp.Popen(command, stdin=sp.PIPE)
    while True:
        if(camera.has_frame()):
            frame = camera.encode_to_png(camera.get_frame_raw())
            process.stdin.write(frame)

def denoised_stream(camera, id):
    sizeStr = camera.get_sizestr()
    fps = camera.get_fps()
    rtsp_server = rtsp_denoised + "/" + id
    command = ['ffmpeg',
            '-re',
            '-s', sizeStr,
            '-r', str(fps),  # rtsp fps (from input server)
            '-i', '-',
               
            # You can change ffmpeg parameter after this item.
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # output fps
            '-g', '50',
            '-c:v', 'libx265',
            '-b:v', '2M',
            '-bufsize', '64M',
            '-maxrate', "4M",
            '-preset', 'veryfast',
            '-rtsp_transport', 'tcp',
            '-segment_times', '5',
            '-f', 'rtsp',
            rtsp_server]

    process = sp.Popen(command, stdin=sp.PIPE)
    while True:
        if(camera.has_frame()):
            frame = camera.encode_to_png(camera.denoise(camera.get_frame_raw()))
            process.stdin.write(frame)

def motion_adaptive_stream(camera, id):
    sizeStr = camera.get_sizestr()
    fps = camera.get_fps()
    rtsp_server = rtsp_motion_adaptive + "/" + id
    command = ['ffmpeg',
            '-fflags', 'nobuffer',
            # '-re',
            '-s', sizeStr,
            # '-r', str(fps),  # rtsp fps (from input server)
            '-i', '-',
               
            # You can change ffmpeg parameter after this item.
            '-pix_fmt', 'yuv420p',
            # '-r', str(fps),  # output fps
            # '-g', '50',
            '-c:v', 'libx265',
            #'-b:v', '2M',
            #'-bufsize', '64M',
            #'-maxrate', "4M",
            #'-preset', 'veryfast',
            '-rtsp_transport', 'tcp',
            # '-segment_times', '5',
            '-f', 'rtsp',
            rtsp_server]

    process = sp.Popen(command, stdin=sp.PIPE)
    while True:
        if(camera.has_frame()):
            frame = camera.encode_to_png(camera.attach_fps(camera.get_frame_raw()))
            time.sleep(1/camera.get_regulated_stream_fps())
            process.stdin.write(frame)

def lowlight_enhance_stream(camera, id):
    sizeStr = camera.get_sizestr()
    fps = camera.get_fps()
    rtsp_server = rtsp_lowlight_enhance
    command = ['ffmpeg',
            '-re',
            '-s', sizeStr,
            '-r', str(fps),  # rtsp fps (from input server)
            '-i', '-',
               
            # You can change ffmpeg parameter after this item.
            '-pix_fmt', 'yuv420p',
            '-r', str(fps),  # output fps
            '-g', '50',
            '-c:v', 'libx264',
            '-b:v', '2M',
            '-bufsize', '64M',
            '-maxrate', "4M",
            '-preset', 'veryfast',
            '-rtsp_transport', 'tcp',
            '-segment_times', '5',
            '-f', 'rtsp',
            rtsp_server]

    process = sp.Popen(command, stdin=sp.PIPE)
    while True:
        if(camera.has_frame()):
            frame = camera.encode_to_png(camera.lowlight_enhance(camera.get_frame_raw()))
            process.stdin.write(frame)

if __name__ == '__main__':
    cameraManager = CameraManager()
    # cameraManager.add_camera("cam0",0,False,60)
    cameraManager.add_camera("cam1","./sample/webcam1.webm",True)
    cameraManager.add_camera("cam2","./sample/webcam2.webm",True)
    cameraManager.add_camera("cam3","./sample/webcam2_clone.webm",True)

    raw_stream_threads = dict()

    for id in cameraManager.get_all_ID():
        raw_stream_threads[id] = threading.Thread(target=raw_stream, args=[cameraManager.get_camera(id),id], daemon=True)
        raw_stream_threads[id].start()
    
        #denoised_stream_thread = threading.Thread(target=denoised_stream, args=[cameraManager.get_camera(id),id], daemon=True)
        #denoised_stream_thread.start()

        # motion_adaptive_stream_thread = threading.Thread(target=motion_adaptive_stream, args=[cameraManager.get_camera(id),id], daemon=True)
        # motion_adaptive_stream_thread.start()
        # lowlight_enhance_stream_thread = threading.Thread(target=lowlight_enhance_stream, args=[cameraManager.get_camera(id),id], daemon=True)
        # lowlight_enhance_stream_thread.start()
    
    while True:
        time.sleep(5000)
        
     

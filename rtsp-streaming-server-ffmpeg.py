import threading
import cv2
import subprocess as sp
from camera import Camera

# dockercommand = ['./rtsp-server-docker']
# sp.Popen(dockercommand,stdin=sp.PIPE)
fps = 60
fps_max = 60

camera = Camera(fps_max)
camera.run()

global thread_raw
global thread_denoise
global thread_motion_adaptive
global thread_lowlight_enhance 
global thread_sample

def raw_stream():
    sizeStr = camera.sizeStr
    rtsp_server = 'rtsp://127.0.0.1:8554/raw_stream'
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
    while(camera.get_frame()!=None):
        frame = camera.encode_to_png(camera.get_frame_raw())
        process.stdin.write(frame.tobytes())

def denoised_stream():
    sizeStr = camera.sizeStr
    rtsp_server = 'rtsp://127.0.0.1:8554/denoised_stream'
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
    while(camera.get_frame()!=None):
        frame = camera.encode_to_png(camera.get_denoised_concat_frame())
        process.stdin.write(frame.tobytes())


def motion_adaptive_stream():
    pass

def lowlight_enhance_stream():
    pass

if __name__ == '__main__':
    raw_stream_thread = threading.Thread(target=raw_stream, daemon=True)
    denoised_stream_thread = threading.Thread(target=raw_stream, daemon=True)
    motion_adaptive_stream_thread = threading.Thread(target=raw_stream, daemon=True)
    lowlight_enhance_stream_thread = threading.Thread(target=raw_stream, daemon=True)
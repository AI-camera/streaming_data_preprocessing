import asyncio
from io import DEFAULT_BUFFER_SIZE
from time import time
from fastapi import FastAPI
from starlette.responses import StreamingResponse
import uvicorn
from camera import Camera
import cv2
import traceback

fps = 24
fps_max = 24
streaming_time = 1000

camera = Camera(fps_max,"./sample/akihabara_01.mp4",True)
camera.run()

app = FastAPI()

async def gen(camera: Camera, regulate_stream_fps=False, get_frame=camera.get_frame):
    counter = 0
    fps = fps_max
    timelist = []
    numframe = 10
    start_stream = time()
    
    while True:
        start = time()
        try:
            frame = None
            # Stop streaming after 60 seconds.
            counter += 1
            if (time() - start_stream) > streaming_time:
                break
            if regulate_stream_fps:
                fps = camera.get_regulated_stream_fps()
                await asyncio.sleep(1/fps)

            frame = get_frame()
            if frame is None:
                print("server-fastapi.py frame is None. Default to error image")
                frame = camera.encode_to_png(camera.default_error_image)
                continue
            else:
                frame = camera.encode_to_png(frame)
            elapsed = time() - start
            # print(f"Encoding time: %.2f" % (finish_encode - start_encode))
            # print(f"Time elapsed per frame: %.2f" % elapsed)
            timelist.append(elapsed)
            if len(timelist) == numframe:
                print(f"Mean elapsed per %d frames: %.2f" % (numframe,sum(timelist)/len(timelist)))
                print(f"Mean FPS: %.2f" % (1/(sum(timelist)/len(timelist))))
                timelist = []

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
        except:
            traceback.print_exc()
        
async def gen_sample():
    counter = 0
    fps = 24
    video = cv2.VideoCapture("./sample/fire1.avi")
    while video.isOpened():
        try:
            counter += 1
            if counter/fps_max > streaming_time:
                break
            if fps < 1:
                fps = 1
            await asyncio.sleep(1/fps)
            ret, frame = video.read()
            if ret == True:
                frame = cv2.imencode('.png', frame)[1].tobytes()
        except Exception as e:
            print(e)
        finally:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.get("motion_FPS")
async def motion_FPS():
    return StreamingResponse(gen(camera, True,get_frame=camera.get_fps_attached_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("denoise")
async def motion_FPS():
    return StreamingResponse(gen(camera, True,get_frame=camera.get_denoised_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("sharpen")
async def motion_FPS():
    return StreamingResponse(gen(camera, True,get_frame=camera.get_sharpened_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("sample")
async def sample():
    return StreamingResponse(gen_sample(), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("lowlight_enhance")
async def lowlight_enhance():
    return StreamingResponse(gen(camera, False, camera.get_lowlight_enhance_concat_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("vehicle_detect")
async def vehicle_detect():
    return StreamingResponse(gen(camera, False, camera.get_detect_object_frame),
    media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("vehicle_detect/lowlight_enhance")
async def vehicle_detect_lowlight_enhance():
    return StreamingResponse(gen(camera, False, camera.get_detect_object_lowlight_enhance_frame), 
    media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/dev")
async def dev():
    return StreamingResponse(gen(camera,camera.get_raw_frame), 
    media_type="multipart/x-mixed-replace; boundary=--frame")

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=5000, log_level="info")

import asyncio
from time import time
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from yolov3_detector import Detector as YOLOV3Detector
import uvicorn
from camera import Camera
import cv2
import traceback

fps = 24
fps_max = 24
streaming_time = 1000
video_source = "./sample/akihabara_03.mp4"
# video_source = 0

camera = Camera(fps_max,video_source,True,detector=YOLOV3Detector(edge_tpu=False))
camera.run()

app = FastAPI()

async def gen(camera: Camera, regulate_stream_fps=False, processing=None):
    if regulate_stream_fps:
        camera.set_regulate_fps_by_motion(True)
    
    while True:
        start = time()
        try:
            frame = camera.get_raw_frame()
            if processing != None:
                frame = processing(frame)

            if frame is None:
                print("server-fastapi.py frame is None. Default to error image")
                frame = camera.encode_to_jpg(camera.default_error_image)
                continue
            else:
                frame = camera.encode_to_jpg(frame)
            # print(f"Encoding time: %.2f" % (finish_encode - start_encode))
            elapsed = time() - start
            # print(f"Time elapsed per frame: %.2f" % elapsed)
            await asyncio.sleep(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
        except:
            # traceback.print_exc()
            break
        
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
            # print(e)
            pass
        finally:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.get("/")
async def root():
    return({"device":"camera_1"})

@app.get("/raw")
async def raw():
    return StreamingResponse(gen(camera), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/motion_FPS")
async def motion_FPS():
    return StreamingResponse(gen(camera, True,processing=camera.attach_motion_text), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/denoise")
async def denoise():
    return StreamingResponse(gen(camera, False,processing=camera.denoise), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/sharpen")
async def sharpen():
    return StreamingResponse(gen(camera, False,processing=camera.sharpen), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/sample")
async def sample():
    return StreamingResponse(gen_sample(), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/lowlight_enhance")
async def lowlight_enhance():
    return StreamingResponse(gen(camera, False, camera.lowlight_enhance), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/vehicle_detect")
async def vehicle_detect():
    # return StreamingResponse(gen(camera, False, camera.detect_object),
    # media_type="multipart/x-mixed-replace; boundary=--frame")
    return {"vehicle_detect_API":"This is down for the moment :D"}

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=3001, log_level="info")

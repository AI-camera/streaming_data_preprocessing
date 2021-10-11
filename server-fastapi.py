import asyncio
from time import time
import numpy as np
# from typing import List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from numpy.lib.function_base import average
# from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from capture import capture_and_save
from camera import Camera
import logging
import logging.config
import conf
import cv2
import timeit

logging.config.dictConfig(conf.dictConfig)
logger = logging.getLogger(__name__)

fps = 24
fps_max = 24
streaming_time = 120

camera = Camera(fps_max)
camera.run()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def entrypoint(request: Request):
    logger.debug("Requested /")
    return templates.TemplateResponse("index.html", {"request": request})


# @app.get("/r")
# async def capture(request: Request):
#     logger.debug("Requested capture")
#     im = camera.get_frame(_byte=False)
#     capture_and_save(im)
#     return templates.TemplateResponse("send_to_init.html", {"request": request})


async def gen(camera, regulate_stream_fps=True, get_frame=camera.get_frame):
    logger.debug("Starting stream")
    counter = 0
    fps = fps_max
    timelist = []
    numframe = 10
    start_stream = time()
    while True:
        start = time()
        try:
            # Stop streaming after 60 seconds.
            counter += 1
            if (time() - start_stream) > streaming_time:
                break
            if regulate_stream_fps:
                fps = camera.get_regulated_stream_fps()
            await asyncio.sleep(1/fps)
            frame = get_frame()
        except Exception as e:
            print(e)
        finally:
            elapsed = time() - start
            # print("Time elapsed per frame" +str(elapsed))
            timelist.append(elapsed)
            if len(timelist) == numframe:
                # print("Mean elapsed per "+ str(numframe) +" frame:" + str(average(timelist)))
                timelist = []

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            # yield (b'--frame\r\n'
            #     b'Content-Type:image/jpeg\r\n'
            #     b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
            #     b'\r\n' + frame + b'\r\n')
        
        

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

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen(camera, False), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed_motion_FPS")
async def video_feed():
    return StreamingResponse(gen(camera, True), media_type="multipart/x-mixed-replace; boundary=--frame")


@app.get("/video_feed_sample")
async def video_feed_sample():
    return StreamingResponse(gen_sample(), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed_denoised")
async def video_feed_denoised():
    return StreamingResponse(gen(camera, False,camera.get_denoised_concat_frame), media_type="multipart/x-mixed-replace; boundary=--frame")


@app.get("/video_feed_he")
async def video_feed_he():
    return StreamingResponse(gen(camera, False, camera.get_he_concat_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed_lowlight_enhance")
async def video_feed_lowlight_enhance():
    return StreamingResponse(gen(camera, False, camera.get_lowlight_enhance_concat_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed_median_blur")
async def video_feed_median_blur():
    return StreamingResponse(gen(camera, False, camera.get_median_blur_concat_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed_viqs")
async def video_feed_viqs():
    return StreamingResponse(gen(camera, False, camera.get_frame_with_motion_points), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed_object_tracking")
async def video_feed_object_tracking():
    return StreamingResponse(gen(camera, False, camera.get_object_tracked_frame), media_type="multipart/x-mixed-replace; boundary=--frame")

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=5000, log_level="info")

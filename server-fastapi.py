import asyncio
from io import DEFAULT_BUFFER_SIZE
from time import time
from fastapi import FastAPI
from starlette.responses import StreamingResponse
import uvicorn
from camera import Camera
import cv2
import traceback

fps = 60
fps_max = 60
streaming_time = 1000

camera = Camera(fps_max,"./sample/akihabara_02.mp4",True)
camera.set_detect_box(0.2,0.4,1,1)
camera.set_selected_classes(["car"])
camera.run()

app = FastAPI(root_path='/cam1/')

async def gen(camera: Camera, detect_motion=False,preprocessing=[], processing=None):
    if detect_motion:
        camera.enable_motion_detect(True)
    camera.set_preprocess_functions(preprocessing)

    while True:
        start = time()
        try:
            frame = camera.get_raw_frame()
            if processing != None:
                # if(detect_motion and camera.motion_detected):
                frame = processing(frame)

            if frame is None:
                print("server-fastapi.py frame is None. Default to error image")
                frame = camera.encode_to_jpg(camera.default_error_image)
                continue
            else:
                frame = camera.encode_to_jpg(frame)
            # print(f"Encoding time: %.2f" % (finish_encode - start_encode))
            elapsed = time() - start
            print(f"Time per frame %.2f, FPS: %.2f" % (elapsed,1/elapsed))
            await asyncio.sleep(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 

        except Exception:
            print(traceback.format_exc())
            break
        # except:
        #     print("something broke")
        #     break

async def gen_sample():
    counter = 0
    fps = 24
    video = cv2.VideoCapture("./sample/face.mp4")
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

@app.get("/")
async def root():
    return({"device":"raspberry_pi"})

@app.get("/raw")
async def raw():
    return StreamingResponse(gen(camera), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/motion_FPS")
async def motion_FPS():
    return StreamingResponse(gen(camera, False,processing=camera.attach_motion_text), media_type="multipart/x-mixed-replace; boundary=--frame")

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
    return StreamingResponse(gen(camera, False, camera.detect_object),
    media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/debug")
async def auto_smart():
    return StreamingResponse(gen(camera,
                                detect_motion=False,
                                preprocessing=[camera.lowlight_enhance],
                                processing=camera.detect_object), 
                            media_type="multipart/x-mixed-replace; boundary=--frame")

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=3000, log_level="info")

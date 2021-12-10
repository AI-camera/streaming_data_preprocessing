from fastapi import FastAPI
import uvicorn
from camera_test import Camera
from starlette.responses import StreamingResponse
from camera_manager import CameraManager

app = FastAPI()

fps_max = 60
cameraManager = CameraManager()

cameraManager.add_camera("cam0",0)
cameraManager.add_camera("cam1","./sample/fire1.avi",True)
cameraManager.add_camera("cam2","./sample/Result-06-10-2021.mp4",True)

async def gen(camera: Camera, regulate_stream_fps=True):
    frame = camera.get_sample_frame_jpg()
    while True:
        try:
            if(camera.has_frame()):
                frame = camera.encode_to_jpg(camera.get_frame_raw())
        except Exception as e:
            print(e)
        finally:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.get("/video0")
async def video0():
    camera = cameraManager.get_camera('cam0')
    return StreamingResponse(gen(camera, False), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video1")
async def video1():
    camera = cameraManager.get_camera('cam1')
    return StreamingResponse(gen(camera, False), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video2")
async def video2():
    camera = cameraManager.get_camera('cam2')
    return StreamingResponse(gen(camera, False), media_type="multipart/x-mixed-replace; boundary=--frame")



if __name__=="__main__":
    uvicorn.run(app, host="localhost", port = 5000, log_level = "info")
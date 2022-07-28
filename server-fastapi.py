import asyncio
from io import DEFAULT_BUFFER_SIZE
from time import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from camera import Camera
import cv2
import traceback

fps = 240
fps_max = 240

# camera = Camera(max_fps = fps_max,video_source=0,allow_loop=False)
# camera = Camera(max_fps = fps_max,video_source="./sample/lowlight_children_02.mp4",allow_loop=True)
# camera = Camera(fps_max,"./sample/akihabara_02.mp4",True)
# camera = Camera(fps_max,"./sample/home_night_01.mp4",True)
# camera = Camera(fps_max,"./sample/face.mp4",True)
# camera = Camera(fps_max,"./sample/video_03.mp4",True)
# camera = Camera(fps_max,"./sample/lowlight_children_01.mp4",True)
camera = Camera(fps_max,"./sample/sample7_children.mp4",True)
camera.set_detect_box(0,0,1,1)
camera.set_frame_skip(0)
camera.set_selected_classes(["people","car"])
camera.run()
camera.lle_pyramid=True 
camera.lle_scale = 1
outer_start = 0

app = FastAPI(root_path='/cam_pi')
predomain = ""

async def gen(camera: Camera, detect_motion=False,preprocessing=[], processing=None):
    # global outer_start

    if detect_motion:
        camera.enable_motion_detect(True)
    # camera.set_preprocess_functions(preprocessing)
    
    while True:
        try:
            # elapsed = time() - outer_start
            # print(f"Time per frame %.2f, FPS: %.2f" % (elapsed,1/elapsed))
            # start = time()
            frame = camera.get_raw_frame()
            for prep in preprocessing:
                frame = prep(frame)
                
            if processing != None:
                frame = cv2.circle(frame, camera.tentative_point_1,2,(255,0,0),5)
                frame = cv2.circle(frame, camera.tentative_point_2,2,(0,0,255),5)
                if(detect_motion and camera.motion_detected):
                    await asyncio.sleep(1)
                if(not detect_motion) or (detect_motion and camera.motion_detected):
                    frame = processing(frame)
                
            if frame is None:
                print("server-fastapi.py frame is None. Default to error image")
                frame = camera.encode_to_jpg(camera.default_error_image)
                continue
            else:
                frame = camera.encode_to_jpg(frame)
            # print(f"Encoding time: %.2f" % (finish_encode - start_encode))
            # elapsed = time() - start
            # print(f"Time per frame %.2f, FPS: %.2f" % (elapsed,1/elapsed))
            await asyncio.sleep(0)
            # outer_start = time()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 
        except asyncio.CancelledError:
            return


# video = cv2.VideoCapture(0) 
# error_frame = cv2.imread("images/500-err.jpg")
async def gen_sample():
    while True:
        try:
            ret, frame = video.read()
            if ret:
                frame = cv2.imencode('.png', frame)[1].tobytes()
            else:
                frame = cv2.imencode('.png', error_frame)[1].tobytes()

        except Exception as e:
            # print(e)
            pass
        finally:
            await asyncio.sleep(0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') 

@app.get("/raw_sample")
async def sample():
    return StreamingResponse(gen_sample(), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get(predomain + "/")
async def root():
    return({"device":"raspberry_pi"})

@app.get(predomain + "/raw")
async def raw():
    try:
        return StreamingResponse(gen(camera), media_type="multipart/x-mixed-replace; boundary=--frame")
    except Exception:
        # print(traceback.format_exc())
        return {"connection":"server singlehandledly closed the connection"}

@app.get(predomain + "/motion_FPS")
async def motion_FPS():
    return StreamingResponse(gen(camera, detect_motion=True ,preprocessing=[camera.attach_motion_text]), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get(predomain + "/denoise")
async def denoise():
    return StreamingResponse(gen(camera, False,preprocessing=[camera.denoise]), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get(predomain + "/sharpen")
async def sharpen():
    return StreamingResponse(gen(camera, False,preprocessing=[camera.sharpen]), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get(predomain + "/sample")
async def sample():
    return StreamingResponse(gen_sample(), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get(predomain + "/lowlight_enhance")
async def lowlight_enhance():
    return StreamingResponse(gen(camera, False, processing=camera.lowlight_enhance), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get(predomain + "/object_detect")
async def object_detect():
    return StreamingResponse(gen(camera, False, processing=camera.detect_object,
    preprocessing=[camera.lowlight_enhance]
    ),
    media_type="multipart/x-mixed-replace; boundary=--frame")

@app.post(predomain + "/set_all_tentative_points_and_confirm")
async def set_all_tentative_points_and_confirm(x1:int, y1:int,x2:int,y2:int):
    x1 = x1/camera.width
    y1 = y1/camera.height
    x2 = x2/camera.width
    y2 = y2/camera.height
    if (x2>x1) and (y2>y1) and (x1>=0) and (y1>=0):
        camera.set_detect_box(x1,y1,x2,y2)
    else:
        raise HTTPException(status_code=400, detail="Irregular tentative points")

@app.post(predomain + "/set_tentative_point_1")
async def set_tentative_point_1(x:int, y:int):
    camera.tentative_point_1 = (x,y)
    
@app.post(predomain + "/set_tentative_point_2")
async def set_tentative_point_2(x:int, y:int):
    camera.tentative_point_2 = (x,y)

@app.post(predomain + "/confirm_detect_box")
async def confirm_detect_box():
    x1,y1 = camera.tentative_point_1
    x2,y2 = camera.tentative_point_2
    
    x1 = x1/camera.width
    y1 = y1/camera.height
    x2 = x2/camera.width
    y2 = y2/camera.height
    if (x2>x1) and (y2>y1) and (x1>=0) and (y1>=0):
        camera.set_detect_box(x1,y1,x2,y2)
    else:
        raise HTTPException(status_code=400, detail="Irregular tentative points!")

@app.post(predomain + "/reset_detect_box")
async def reset_detect_box():
    camera.tentative_point_1 = (-10,-10)
    camera.tentative_point_2 = (-10,-10)
    camera.set_detect_box(0,0,1,1)

@app.post(predomain + "/reset_tentative_points")
async def reset_tentative_points():
    camera.tentative_point_1 = (-10,-10)
    camera.tentative_point_2 = (-10,-10)

@app.get(predomain + "/get_camera_size")
async def get_camera_size():
    return [camera.width, camera.height]

@app.post(predomain + "/add_detect_class")
async def add_detect_class(class_name:str):
    if class_name not in camera.detector.labels:
        raise HTTPException(status_code=400, detail="Unavailable classname!")
    elif class_name not in camera.selected_classes:
        camera.selected_classes.append(class_name)

@app.post(predomain + "/reset_detect_class")
async def reset_detect_class():
    camera.selected_classes = ["None"]

@app.post(predomain + "/add_device_token")
async def add_device_token(token:str):
    if token not in camera.firebase_tokens:
        camera.firebase_tokens.append(token)

@app.post(predomain + "/remove_device_token")
async def remove_device_token(token:str):
    if token in camera.firebase_tokens:
        camera.firebase_tokens.remove(token)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=3003, log_level="info")

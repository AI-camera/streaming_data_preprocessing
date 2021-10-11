import lowlight_enhance
import cv2
import numpy as np
import threading
import time
import logging
from object_tracking.optical_flow_motion_detector import OpticalFlowMotionDetector
from object_tracking.object_tracking import ObjectTracking

logger = logging.getLogger(__name__)

thread = None


class Camera:
    def __init__(self, fps=20, video_source=0):
        logger.info(
            f"Initializing camera class with {fps} fps and video_source={video_source}")
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        # We want a max of 5s history to be stored, thats 5s*fps
        self.max_frames = 5*self.fps
        self.frames = []
        self.isrunning = False
        self.stream_fps = fps
        self.tick = 0
        self.fps_lock = False
        self.motion_detector = None
        self.start_time = time.time()
        self.end_tine = time.time()
        self.object_tracker = None
        self.sizeStr = str(int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))) + 'x' + str(int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def run(self):
        logging.debug("Perparing thread")
        global thread
        if thread is None:
            logging.debug("Creating thread")
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            logger.debug("Starting thread")
            self.isrunning = True
            thread.start()
            logger.info("Thread started")

    def _capture_loop(self):
        dt = 1/self.fps
        logger.debug("Observation started")

        v, img = self.camera.read()
        self.first_frame_initialize(img)

        while self.isrunning:
            v, img = self.camera.read()
            if v:
                if len(self.frames) == self.max_frames:
                    self.frames = self.frames[1:]
                self.frames.append(img)
            time.sleep(dt)
            self.regulate_stream_fps()
        logger.info("Thread stopped successfully")

    def first_frame_initialize(self,first_frame):
        self.motion_detector = OpticalFlowMotionDetector(first_frame)
        self.object_tracker = ObjectTracking(first_frame)

    def stop(self):
        logger.debug("Stopping thread")
        self.isrunning = False

    def attach_fps(self, frame):
        return cv2.putText(frame, 'FPS: ' + str(self.get_regulated_stream_fps()), (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

    def encode_to_png(self, frame):
        return cv2.imencode('.png', frame)[1].tobytes()

    def encode_to_jpg(self, frame):
        return cv2.imencode('.jpg', frame)[1].tobytes()

    def get_frame(self, _bytes=True):
        if len(self.frames) > 0:
            frame_with_fps = self.attach_fps(self.get_frame_raw())
            if _bytes:
                img = self.encode_to_jpg(frame_with_fps)
            else:
                img = frame_with_fps
        else:
            with open("images/not_found.jpeg", "rb") as f:
                img = f.read()
        return img

    def get_frame_raw(self):
        if len(self.frames) > 0:
            return self.frames[-1]

    def get_sizestr(self):
        return self.sizeStr

    def regulate_stream_fps(self):
        locktime = 2
        try:
            if(self.fps_lock and (time.time() - self.start_time) > locktime):
                self.fps_lock = False
            frame_raw = self.get_frame_raw()
            fps_adjustment = np.ceil(
                self.motion_detector.detect(frame_raw) % self.fps)
            if(not self.fps_lock):
                self.stream_fps = fps_adjustment

            # Bump the fps up if there's motion
            # print(fps_adjustment)
            if(fps_adjustment > 3):
                self.stream_fps = self.fps
                self.fps_lock = True
                self.start_time = time.time()
        except Exception as e:
            print(e)

    def get_regulated_stream_fps(self):
        if self.stream_fps < 1:
            return 1
        else:
            return self.stream_fps

    #Denoise using bilateral filter
    def denoise(self, frame):
        denoised_frame = None
        if(frame is None):
            print("Denoising None frame")
            return None
        try:
            # denoised_frame = cv2.fastNlMeansDenoisingColored(frame,None)
            denoised_frame = cv2.bilateralFilter(frame, 5, 75, 75)
        except Exception as e:
            print(e)
        finally:
            return denoised_frame

    def get_denoised_concat_frame(self):
        return self.get_concat_frame(self.denoise)

    def median_blur(self,frame):
        return cv2.medianBlur(frame,3)

    def get_median_blur_concat_frame(self):
        return self.get_concat_frame(self.median_blur)    

    #Histogram Equalize
    def he(self, frame):
        frame_YCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        channels = cv2.split(frame_YCC)
        channels[0] = cv2.equalizeHist(channels[0])
        frame_YCC = cv2.merge((channels[0], channels[1], channels[2]))
        return cv2.cvtColor(frame_YCC, cv2.COLOR_YUV2BGR)

    def get_he_concat_frame(self):
        return self.get_concat_frame(self.he)

    #Low light enhance
    def lowlight_enhance(self, frame):
        return lowlight_enhance.lowlight_enhance(frame)

    def get_lowlight_enhance_concat_frame(self):
        return self.get_concat_frame(self.lowlight_enhance)
    #Auto VIQS
    ##Using Optical Flow 
    ###Detect motion points, draw bounding box around those points, reduce img quality outside those boxes
    def attach_motion_points(self,frame):
        bbox_width = 40
        self.motion_detector.refresh_motion_points(frame)
        for i, (new,old) in self.motion_detector.get_motion_points():
            a,b = old.ravel()
            c,d = new.ravel()
            motion_value = np.sqrt((a-c)**2 + (b-d)**2)
            if(motion_value > -1 and motion_value <10):
                top_left = (int(c-bbox_width),int(d-bbox_width))
                bottom_right = (int(c+bbox_width),int(d+bbox_width))
                color = (255,255,0)
                # frame = cv2.rectangle(frame,top_left,bottom_right,color,thickness=1)
                frame = cv2.circle(frame,(c,d),radius=3,color=(255,255,0),thickness=-1)

    def get_frame_with_motion_points(self):
        frame = self.get_frame_raw()
        if(self.motion_detector is not None):
            self.attach_motion_points(frame)

        return self.encode_to_jpg(frame)

    def viqs(self,frame):
        pass

    def get_viqs_concat_frame(self):
        return self.get_concat_frame(self.viqs)

    ##Using Opencv trackers
    ###Draw bounding box around
    def track_object(self,frame):
        return self.object_tracker.track(frame)

    def get_object_tracked_frame(self):
        return self.get_single_frame(self.track_object)

    def get_single_frame(self,preprocessfunc):
        raw_frame = self.attach_fps(self.get_frame_raw())
        preprocessed_frame = self.attach_fps(preprocessfunc(raw_frame))
        return preprocessed_frame

    def get_concat_frame(self, preprocessfunc):
        raw_frame = self.get_frame_raw()
        preprocessed_frame = preprocessfunc(raw_frame)
        frame = np.concatenate((raw_frame, preprocessed_frame), axis=1)
        return frame
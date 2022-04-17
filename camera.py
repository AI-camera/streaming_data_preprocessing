from os import stat_result, truncate

from cv2 import COLOR_BGR2HSV
import dehaze
import cv2
import numpy as np
import threading
import time
import random
from object_tracking.motion_detector import OpticalFlowMotionDetector
from track import Track
from yolov3_detector import Detector as YOLOV3Detector
from mobileDet_detector import Detector as MobileDetDetector
from sort import *
from utils import *
from track_manager import TrackManager
from track import Track
thread = None

VIDEO_05_MARKERLINES = dict()
VIDEO_05_MARKERLINES["L1"] = ((0.28,0.68),(0.54,0.57))
VIDEO_05_MARKERLINES["L2"] = ((0.62,0.55),(0.93,0.57))
VIDEO_05_MARKERLINES["L3"] = ((0.98,0.58),(0.94,0.82))
VIDEO_05_MARKERLINES["L4"] = ((0.29,0.73),(0.89,0.88))
VIDEO_05_DETECT_BOX = ((0.2,0.4),(1,1))
VIDEO_05_REDLIGHT_BOX = ((0.972,0.27),(0.984,0.33))

AKIHABARA_01_MARKERLINES = dict()
AKIHABARA_01_MARKERLINES["L1"] = ((0.28,0.68),(0.54,0.57))
AKIHABARA_01_MARKERLINES["L2"] = ((0.62,0.55),(0.93,0.57))
AKIHABARA_01_MARKERLINES["L3"] = ((0.98,0.58),(0.94,0.82))
AKIHABARA_01_MARKERLINES["L4"] = ((0.29,0.73),(0.89,0.88))
AKIHABARA_01_CROPBOX = ((0.2,0.4),(1,1))
MOTION_THRESHOLD = 3

JACKSONHOLE_SNOWY_REDLIGHT_BOX = ((0.962,0.27),(0.974,0.33))
FOG_TRAFLIGHT_REDLIGHT_BOX = ((0.794,0.428),(0.81,0.452))
AKIHABARA_03_REDLIGHT_BOX = ((0.101,0.21),(0.12,0.240))

class Camera:
    def __init__(self, fps=24, video_source=0, allow_loop=False, detector = MobileDetDetector(), markerlines = VIDEO_05_MARKERLINES):
        """
        - fps: Rate at which frames are read from video_source
        - video_source: The video_source to read frames from. Defaulted to 0 (webcam). Anything that can be used in cv2.VideoCapture
        - allow_loop: Set to True to allow looping on video files. This turns those files into endless stream
        """
        self.fps = fps
        self.video_source = cv2.VideoCapture(video_source)
        # We want a max of 1s history to be stored, thats 3s*fps
        self.max_frames = 1*self.fps
        self.frames = []
        self.lowlight_enhanced_frames = []
        self.isrunning = False
        self.stream_fps = fps
        self.tick = 0
        self.fps_lock = False
        self.motion_detector = None
        self.start_time = time.time()
        self.end_tine = time.time()

        self.sizeStr = str(int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))) + 'x' + str(int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.sizeStrConcat = str(int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH)*2)) + 'x' + str(int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.allow_loop = allow_loop

        self.detector = detector

        self.default_error_image = cv2.imread("images/500-err.jpg")

        # Optical Flow motion detector
        self.motion_detector = OpticalFlowMotionDetector()

        # SORT tracker
        self.tracker = Sort()
        self.trackManager = TrackManager()

        # Marker lines dictionary
        self.markerlines_dict = markerlines

        # Define video file output
        width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_source.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = cv2.VideoWriter("./output/output.avi", codec, fps, (width, height))

        # Define detect boxes
        self.detect_box = ((0,0),(1,1))

        # Define redlight boxes
        self.redlight_box = ((0,0),(1,1))

        # Define detection properties
        self.skipped_frame_count = -1
        self.last_frame_output = None
        self.frameskip = 0
        self.selected_classes = ["car","motorbike"]
        self.preprocess_functions = []
        self.redlight_markerline_ids=[]
        
        self.run()

    def set_detect_box(self,detect_box):
        self.detect_box = detect_box

    def set_frame_skip(self,frameskip):
        self.frameskip = frameskip

    def set_selected_classes(self,selected_classes):
        self.selected_classes = selected_classes

    def set_preprocess_functions(self,preprocess_functions):
        self.preprocess_functions = preprocess_functions
    
    def set_redlight_markerline_ids(self, redlight_markerline_ids):
        self.redlight_markerline_ids = redlight_markerline_ids

    def set_marker_lines(self,markerlines):
        self.markerlines_dict = markerlines

    def run(self):
        global thread
        global subthread1
        thread = threading.Thread(target=self._capture_loop,daemon=True)
        subthread1 = threading.Thread(target=self._lowlight_enhance_loop,daemon=True)
        if not self.isrunning:
            self.isrunning = True
            thread.start()
            # subthread1.start()
        else:
            print("A camera thread is running already!")

    def _capture_loop(self):
        dt = 1/self.fps
        v, img = self.video_source.read()

        while self.isrunning:
            v, img = self.video_source.read()
            if v:
                if len(self.frames) >= 2:
                    self.frames = self.frames[1:]
                self.frames.append(img)
            elif(self.allow_loop):
                print("camera.py  End of Video. Loop from start")
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            time.sleep(dt)
            self.regulate_stream_fps()
    
    def _lowlight_enhance_loop(self):
        while self.isrunning:
            frame = self.get_raw_frame()
            if frame is not None:
                if len(self.lowlight_enhanced_frames) == self.max_frames:
                    self.lowlight_enhanced_frames = self.lowlight_enhanced_frames[1:]
                frame = self.lowlight_enhance(frame)
                self.lowlight_enhanced_frames.append(frame)

    def stop(self):
        self.isrunning = False

    def attach_fps(self, frame):
        return cv2.putText(frame, 'FPS: ' + str(self.get_regulated_stream_fps()), (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

    def get_fps_attached_frame(self):
        return self.attach_fps(self.get_raw_frame())

    def encode_to_png(self, frame):
        return cv2.imencode('.png', frame)[1].tobytes()

    def encode_to_jpg(self, frame):
        return cv2.imencode('.jpg', frame)[1].tobytes()

    def get_frame(self, _bytes=True):
        if len(self.frames) > 0:
            frame_with_fps = self.attach_fps(self.get_raw_frame())
            if _bytes:
                img = self.encode_to_jpg(frame_with_fps)
            else:
                img = frame_with_fps
        else:
            with open("images/not_found.jpeg", "rb") as f:
                img = f.read()
        return img

    def get_sample_frame_jpg(self):
        return self.encode_to_jpg(cv2.imread("./images/not_found.jpeg"))

    def get_raw_frame(self):
        if len(self.frames) > 0:
            return self.frames[0]
    
    def has_frame(self):
        if len(self.frames) > 0:
            return True
        return False

    def get_fps(self):
        return self.fps
        
    def get_sizestr(self):
        return self.sizeStr

    def get_sizestrConcat(self):
        return self.sizeStrConcat

    def regulate_stream_fps(self):
        locktime = 2
        try:
            if(self.fps_lock and (time.time() - self.start_time) > locktime):
                self.fps_lock = False
            frame_raw = self.get_raw_frame()
            fps_adjustment = np.ceil(
                self.motion_detector.detect_optical_flow(frame_raw) % self.fps)
            if(not self.fps_lock):
                self.stream_fps = fps_adjustment

            # Bump the fps up if there's motion
            # print(fps_adjustment)
            if(fps_adjustment > MOTION_THRESHOLD):
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
   
    def denoise(self, frame):
        '''
        - Denoise using median filter
        '''
        denoised_frame = None
        if(frame is None):
            return None
        try:
            # denoised_frame = cv2.fastNlMeansDenoisingColored(frame,None)
            denoised_frame = cv2.medianBlur(frame,3)
        except Exception as e:
            print(e)
        finally:
            return denoised_frame

    def get_denoised_frame(self,frame):
        return self.denoise(self.get_raw_frame())

    def get_denoised_concat_frame(self):
        return self.get_concat_frame(self.denoise)

    def median_blur(self,frame):
        return cv2.medianBlur(frame,3)

    def gaussian_blur(self,frame):
        return cv2.GaussianBlur(frame,(5,5),0)
    
    def sharpen(self,frame):
        return 2*frame - self.gaussian_blur(frame)

    def get_sharpened_frame(self):
        return self.sharpen(self.get_raw_frame())

    def get_median_blur_concat_frame(self):
        return self.get_concat_frame(self.median_blur)    

    def he(self, frame):
        '''
        - Histogram Equalize
        '''
        frame_YCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        channels = cv2.split(frame_YCC)
        channels[0] = cv2.equalizeHist(channels[0])
        frame_YCC = cv2.merge((channels[0], channels[1], channels[2]))
        return cv2.cvtColor(frame_YCC, cv2.COLOR_YUV2BGR)

    def get_he_concat_frame(self):
        return self.get_concat_frame(self.he)

    def lowlight_enhance(self, img, scale = 1):
        '''
        - Lowlight enhance
        '''
        #Downscale the image for better performance
        if scale == 1:
            return dehaze.lowlight_enhance(img)

        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = dehaze.lowlight_enhance(img)

        scale = 1/scale #Scale back
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return result.astype('uint8')
    
    def dehaze(self, img, scale = 1):
        '''
        - Lowlight enhance
        '''
        #Downscale the image for better performance
        if scale == 1:
            return dehaze.dehaze(img)

        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = dehaze.dehaze(img)

        scale = 1/scale #Scale back
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return result

    def get_lowlight_enhanced_frame_from_buffer(self):
        if len(self.lowlight_enhanced_frames) > 0:
            return self.lowlight_enhanced_frames[-1]

    def get_lowlight_enhance_concat_frame(self):
        return self.get_concat_frame(self.lowlight_enhance)
    
    def attach_motion_points(self,frame):
        '''
        - Using Optical Flow 
        - Detect motion points, draw bounding box around those points, reduce img quality outside those boxes
        '''
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
        frame = self.get_raw_frame()
        if(self.motion_detector is not None):
            self.attach_motion_points(frame)

        return self.encode_to_jpg(frame)

    def get_single_frame(self,preprocessfunc):
        raw_frame = self.attach_fps(self.get_raw_frame())
        preprocessed_frame = self.attach_fps(preprocessfunc(raw_frame))
        return preprocessed_frame

    def get_concat_frame(self, preprocessfunc):
        raw_frame = self.get_raw_frame()
        preprocessed_frame = preprocessfunc(raw_frame)
        frame = np.concatenate((raw_frame, preprocessed_frame), axis=1)
        return frame
    
    def sp_noise(self,image,prob):
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = np.floor(255*(1-rdn))
                elif rdn > thres:
                    output[i][j] = np.floor(255*rdn)
                else:
                    output[i][j] = image[i][j]
        return output

    def track_object(self,bbs):
        '''
        - Using SORT trackers
        - Draw bounding box around
        - Args:
            * bbs: A list of bounding boxes in format (x1,y1,x2,y2)
        - Return:
            * [bounding_boxes, tracker_id]
        '''
        for bb in bbs:
            bb = np.append(bb,1)
        track_bbs_ids = self.tracker.update(bbs)
        # print("camera.py track_object() ",track_bbs_ids)
        return track_bbs_ids
    
    def draw_tracks(self,frame):
        '''
            - Draw all tracks in trackManager onto frame
            - Args:
                * frame: the frame that is drawn onto
        '''
        for track in self.trackManager.tracks:
            frame = cv2.circle(frame, track.GetCurrentPosition(), radius=2, color=(0, 0, 255), thickness=-1)
        return frame

    def detect_object(self, frame):
        '''
        Detect chosen object from frame
        '''
        # frame = frame.copy()
        x1,y1 = denormalize_coordinate(frame,self.detect_box[0])
        x2,y2 = denormalize_coordinate(frame,self.detect_box[1])

        # Run inference, get boxes
        # Skip *frame_skip* frames per 1 infered frame
        if self.skipped_frame_count<self.frame_skip and self.skipped_frame_count>=0:
            boxes, scores, pred_classes = self.last_frame_output
            self.skipped_frame_count +=1
        else:
            boxes, scores, pred_classes = self.detector.detect(frame, self.detect_box, self.frame_skip, selected_classes = self.selected_classes, preprocess_functions = preprocess_functions)
            self.last_frame_output = (boxes, scores, pred_classes)
            self.skipped_frame_count = 0

        frame = self.draw_marker_lines(frame)
        frame = self.draw_detect_box(frame,(x1,y1,x2,y2))

        boxes_to_track = []
        tracked_boxes_and_ids = []
        for box in boxes:
            (x1_box,y1_box),(x2_box,y2_box) = box
            #Put offset x1 and y1 to match crop box
            boxes_to_track.append([x1_box+x1,y1_box+y1,x2_box+x1,y2_box+y1])

        if len(boxes_to_track) > 0:
            tracked_boxes_and_ids = self.track_object(boxes_to_track)

        denormalized_markerlines_dict = self.markerlines_dict.copy()
        for key in denormalized_markerlines_dict.keys():
            coordinate1 = denormalized_markerlines_dict[key][0]
            coordinate2 = denormalized_markerlines_dict[key][1]
            coordinate1 = denormalize_coordinate(frame, coordinate1)
            coordinate2 = denormalize_coordinate(frame, coordinate2)
            denormalized_markerlines_dict[key] = (coordinate1, coordinate2)

        self.trackManager.HandleNewTracks(tracked_boxes_and_ids,denormalized_markerlines_dict, self.redlight_markerline_ids)

        #Draw all the tracks   
        for track in self.trackManager.tracks:
            if track.isActive :
                track_x, track_y = track.GetCurrentPosition()
                frame = cv2.circle(frame,(track_x, track_y), radius=2, color=(0, 0, 255), thickness=-1) 
                
                text = 'Crossed: '
                for markerline in track.crossedMarkerlineIDs:
                    text += markerline
                    text += ','
                cv2.putText(frame, text, (track_x, track_y), cv2.FONT_HERSHEY_DUPLEX,0.45, (0,255,0), 1, cv2.LINE_AA)
                
                # Connect track's history dots
                for i in range(len(track.history)-1):
                    history1 = track.history[i]
                    history2 = track.history[i+1]
                    history1_x, history1_y = history1[0],history1[1]
                    history2_x, history2_y = history2[0],history2[1]
                    frame = cv2.line(frame,(history1_x,history1_y),(history2_x,history2_y),color=(255, 0, 0),thickness=2)
        
        self.draw_marker_lines(frame)

        if len(boxes) > 0:
            frame = self.detector.draw_boxes(frame, boxes, scores, pred_classes, x1, y1)

        if len(self.redlight_markerline_ids) > 0:
            text = f"RED LIGHT!"
            textpos=(0,100)
            cv2.putText(frame, text, textpos, cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1, cv2.LINE_AA)
        
        return frame
    
    def detect_object_lowlight_enhance(self, frame):
        self.set_preprocess_functions = [self.lowlight_enhance]
        return self.detect_object(frame)

    def get_detect_object_lowlight_enhance_frame(self):
        frame = self.get_raw_frame()
        if frame is not None:
            return self.detect_object_lowlight_enhance(frame)

    def get_detect_object_frame(self):
        frame = self.get_raw_frame()
        if frame is not None:
            return self.detect_object(frame)

    def draw_marker_lines(self,frame):
        '''Loop over the marker line dictionary to get each marker line'''
        color=(0, 255, 255)
        for key in self.markerlines_dict:
            markerline = self.markerlines_dict[key]
            x1,y1 = denormalize_coordinate(frame, markerline[0])
            x2,y2 = denormalize_coordinate(frame, markerline[1])
            frame = cv2.line(frame,(x1,y1),(x2,y2),color=color,thickness=2)
            frame = cv2.putText(frame, key, (x1,y1-3), cv2.FONT_HERSHEY_DUPLEX,
                    0.45, color, 1, cv2.LINE_AA)
        return frame
    
    def draw_detect_box(self,image,box):
        '''
        - Draw the rectangular crop region in the image
        - Args:
            * image
            * x1, x2, y1, y2: Coordinate of topleft and botright of the crop region
        '''
        ((x1,y1),(x2,y2)) = box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        return image

    def write_frame_to_output_file(self,frame):
        '''Write frame to out.mp4'''
        self.out.write(frame)

    def detect_red_light(self,frame):
        '''Detect if the traffic light is green or red depending on the number of green and red pixels in the frame
        @params[frame]: frame in bgr format 
        '''
        
        if frame is None:
            return False
        new_frame = cv2.cvtColor(frame,COLOR_BGR2HSV)
        pixel_count = 0
        red_count = 0
        for row in new_frame:
            for pixel in row:
                pixel_count += 1
                h,s,v = pixel
                if h <= 10:
                    red_count += 1

        # print("Red density is: ")
        # print(red_count/pixel_count)
        if red_count/pixel_count > 0.1:
            return True

if __name__ == '__main__':
    fps = 24
    fps_max = 24

    camera = Camera(fps_max,"./sample/akihabara_03.mp4",True, detector=MobileDetDetector())
    camera.set_marker_lines(VIDEO_05_MARKERLINES)
    camera.set_detect_box(VIDEO_05_DETECT_BOX)
    camera.set_frame_skip(3)
    camera.run()
    i = 0
    inference_time_history = []
    while True:
        frame = camera.get_raw_frame()
        if frame is None:
            continue           

        i +=1
        if i > 2000:
            break
            
        start = time.time()

        # Brightness enhancing
        # print("Brightness before enhancing")
        # print("%.2f" % brightness(frame))
        # print("Brightness after enhancing")
        # print("%.2f" % brightness(frame))

        # Retrieve redlight_box and denormalize it
        redlight_box = denormalize_coordinates(frame,AKIHABARA_03_REDLIGHT_BOX)

        # Detect redlight and add ids to redlight markerline ids
        redlight_crop = imcrop(frame, redlight_box)
        if camera.detect_red_light(redlight_crop):
            camera.redlight_markerline_ids = ["wee"]
        else:
            camera.redlight_markerline_ids = []

        # Write "Red Light" on frame if the light is red
        if len(camera.redlight_markerline_ids) > 0:
            text = f"RED LIGHT!"
            textpos=(0,100)
            cv2.putText(frame, text, textpos, cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1, cv2.LINE_AA)
        

        # Running time calculation
        inference_time_history.append(time.time() - start)
        if len(inference_time_history) > 10:
            inference_time_history = inference_time_history[1:]

        if i%100 == 0 and i > 0:
            print("camera.py executing " + str(i))
            print("FPS: ", str(1/(sum(inference_time_history)/len(inference_time_history))))

        # camera.draw_detect_box(frame,redlight_box)
        # Write camera to  out.png
        cv2.imwrite("out.png",frame)

        camera.write_frame_to_output_file(frame)

    camera.out.release()
    

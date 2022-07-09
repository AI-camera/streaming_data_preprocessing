from os import stat_result, truncate

from cv2 import COLOR_BGR2HSV
import dcp_dehaze
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
from frame_differencing import frame_diff
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
    def __init__(self, max_fps=24, video_source=0, allow_loop=False, detector = MobileDetDetector(), markerlines = {}):
        """
        - fps: Rate at which frames are read from video_source
        - video_source: The video_source to read frames from. Defaulted to 0 (webcam). Anything that can be used in cv2.VideoCapture
        - allow_loop: Set to True to allow looping on video files. This turns those files into endless stream
        """
        self.fps = max_fps
        self.max_fps = max_fps
        self.video_source = cv2.VideoCapture(video_source)
        # We want a max of 1s history to be stored, thats 3s*fps
        self.max_frames = 1*self.fps
        self.frames = []
        self.lowlight_enhanced_frames = []
        self.isrunning = False
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

        # motion detector properties
        # self.motion_detector = OpticalFlowMotionDetector()
        self.last_frame = None
        self.motion_detect_enabled = False
        self.motion_detected = False
        self.motion_lock_counter = 0
        self.motion_detect_skipped_frame=-1
        self.motion_detect_frame_skips=5

        # SORT tracker
        self.tracker = Sort()
        self.trackManager = TrackManager()

        # Marker lines dictionary
        self.markerlines_dict = markerlines

        # Define video file output
        self.width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_source.get(cv2.CAP_PROP_FPS))
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = cv2.VideoWriter("./output/output.avi", self.codec, self.fps, (self.width, self.height))

        # Define detect boxes
        self.detect_box = ((0,0),(1,1))

        # Define redlight boxes
        self.redlight_box = ((0,0),(1,1))

        # Define detection properties
        self.skipped_frame_count = -1
        self.last_frame_output = None
        self.frame_skip = 0
        self.selected_classes = ["car","motorbike"]
        # self.preprocess_functions = []
        self.redlight_markerline_ids=[]
        
        # self.run()

    def set_detect_box(self,x1,y1,x2,y2):
        '''The detect_box should be percentage'''
        self.detect_box = ((x1,y1),(x2,y2))

    def set_frame_skip(self,frameskip):
        '''Set how many frame is skipped before a frame is inferenced by model'''
        self.frame_skip = frameskip

    def set_selected_classes(self,selected_classes):
        '''Set list classes to be detected'''
        self.selected_classes = selected_classes

    def set_preprocess_functions(self,preprocess_functions):
        '''Must set a list of preprocess_functions'''
        self.detector.set_preprocess_functions(preprocess_functions)
    
    def set_redlight_markerline_ids(self, redlight_markerline_ids):
        self.redlight_markerline_ids = redlight_markerline_ids

    def set_video_output(self, output):
        self.out = cv2.VideoWriter(output, self.codec, self.fps, (self.width, self.height))

    def set_marker_lines(self,markerlines):
        self.markerlines_dict = markerlines

    def enable_motion_detect(self,motion_detect_enabled):
        self.motion_detect_enabled = motion_detect_enabled

    def run(self):
        global thread
        global subthread1
        thread = threading.Thread(target=self._capture_loop,daemon=True)
        # subthread1 = threading.Thread(target=self._lowlight_enhance_loop,daemon=True)
        if not self.isrunning:
            self.isrunning = True
            thread.start()
            # subthread1.start()
        else:
            print("A camera thread is running already!")

    def _capture_loop(self):
        # dt = 1/self.fps
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
            
            if (self.motion_detect_enabled):
                # self.detect_motion()
                if (self.motion_detect_skipped_frame >= self.motion_detect_frame_skips):
                    self.detect_motion()
                    self.motion_detect_skipped_frame = 0
                else:
                    self.motion_detect_skipped_frame += 1

            # time.sleep(dt)
    
    # def _lowlight_enhance_loop(self):
    #     while self.isrunning:
    #         frame = self.get_raw_frame()
    #         if frame is not None:
    #             if len(self.lowlight_enhanced_frames) == self.max_frames:
    #                 self.lowlight_enhanced_frames = self.lowlight_enhanced_frames[1:]
    #             frame = self.lowlight_enhance(frame)
    #             self.lowlight_enhanced_frames.append(frame)

    def stop(self):
        self.isrunning = False

    def attach_fps(self, frame):
        return cv2.putText(frame, 'FPS: ' + str(self.get_fps()), (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

    def attach_motion_text(self, frame):
        new_frame = frame.copy()
        return cv2.putText(new_frame, 'motion: ' + str(self.motion_detected), (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

    def encode_to_png(self, frame):
        return cv2.imencode('.png', frame)[1].tobytes()

    def encode_to_jpg(self, frame):
        return cv2.imencode('.jpg', frame)[1].tobytes()

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

    def detect_motion(self):
        start = time.time()
        current_frame = self.get_raw_frame()
        if self.last_frame is None:
            self.last_frame = current_frame
            return 

        if current_frame is not None:
            resized_frame = cv2.resize(current_frame, (300,300), interpolation = cv2.INTER_AREA)
            if self.motion_lock_counter > 0:
                self.motion_lock_counter -= 1
                self.current_frame = None
            elif(frame_diff(self.last_frame, current_frame)):
                self.motion_detected = True
                self.motion_lock_counter = 10
            else:
                self.motion_detected = False
                self.current_frame = None
                
        
        self.last_frame = current_frame
        # print("motion_detect_time:")
        # print(time.time()-start)

    def get_fps(self):
        return self.fps
   
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

    def median_blur(self,frame):
        return cv2.medianBlur(frame,3)

    def gaussian_blur(self,frame):
        return cv2.GaussianBlur(frame,(5,5),0)
    
    def sharpen(self,frame):
        return 2*frame - self.gaussian_blur(frame)

    def he(self, frame):
        '''
        - Histogram Equalize
        '''
        frame_YCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        channels = cv2.split(frame_YCC)
        channels[0] = cv2.equalizeHist(channels[0])
        frame_YCC = cv2.merge((channels[0], channels[1], channels[2]))
        return cv2.cvtColor(frame_YCC, cv2.COLOR_YUV2BGR)

    def lowlight_enhance(self, img, scale = 1):
        '''
        - Lowlight enhance
        '''
        #Downscale the image for better performance
        if scale == 1:
            return dcp_dehaze.lowlight_enhance(img)

        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = dcp_dehaze.lowlight_enhance(img)

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
            return dcp_dehaze.dehaze(img)

        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = dcp_dehaze.dehaze(img)

        scale = 1/scale #Scale back
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        result = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        return result
    
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
            # start = time.time()
            boxes, scores, pred_classes = self.detector.detect(frame, self.detect_box, selected_classes = self.selected_classes)
            # print("detect time: %.2f" % (time.time() - start))
            self.last_frame_output = (boxes, scores, pred_classes)
            self.skipped_frame_count = 0

        # frame = self.draw_marker_lines(frame)
        frame = self.draw_detect_box(frame,((x1,y1),(x2,y2)))

        # boxes_to_track = []
        # tracked_boxes_and_ids = []
        # for box in boxes:
        #     (x1_box,y1_box),(x2_box,y2_box) = box
        #     #Put offset x1 and y1 to match crop box
        #     boxes_to_track.append([x1_box+x1,y1_box+y1,x2_box+x1,y2_box+y1])

        # if len(boxes_to_track) > 0:
        #     tracked_boxes_and_ids = self.track_object(boxes_to_track)

        # denormalized_markerlines_dict = self.markerlines_dict.copy()
        # for key in denormalized_markerlines_dict.keys():
        #     coordinate1 = denormalized_markerlines_dict[key][0]
        #     coordinate2 = denormalized_markerlines_dict[key][1]
        #     coordinate1 = denormalize_coordinate(frame, coordinate1)
        #     coordinate2 = denormalize_coordinate(frame, coordinate2)
        #     denormalized_markerlines_dict[key] = (coordinate1, coordinate2)

        # self.trackManager.HandleNewTracks(tracked_boxes_and_ids,denormalized_markerlines_dict, self.redlight_markerline_ids)

        #Draw all the tracks   
        # for track in self.trackManager.tracks:
        #     if track.isActive :
        #         track_x, track_y = track.GetCurrentPosition()
        #         frame = cv2.circle(frame,(track_x, track_y), radius=2, color=(0, 0, 255), thickness=-1) 
                
        #         text = 'Crossed: '
        #         for markerline in track.crossedMarkerlineIDs:
        #             text += markerline
        #             text += ','
        #         cv2.putText(frame, text, (track_x, track_y), cv2.FONT_HERSHEY_DUPLEX,0.45, (0,255,0), 1, cv2.LINE_AA)
                
        #         # Connect track's history dots
        #         for i in range(len(track.history)-1):
        #             history1 = track.history[i]
        #             history2 = track.history[i+1]
        #             history1_x, history1_y = history1[0],history1[1]
        #             history2_x, history2_y = history2[0],history2[1]
        #             frame = cv2.line(frame,(history1_x,history1_y),(history2_x,history2_y),color=(255, 0, 0),thickness=2)
        
        # self.draw_marker_lines(frame)

        if len(boxes) > 0:
            frame = self.detector.draw_boxes(frame, boxes, scores, pred_classes, x1, y1)

        # if len(self.redlight_markerline_ids) > 0:
        #     text = f"RED LIGHT!"
        #     textpos=(0,100)
        #     cv2.putText(frame, text, textpos, cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255), 1, cv2.LINE_AA)
        
        return frame
    
    # def detect_object_lowlight_enhance(self, frame):
    #     self.set_preprocess_functions = [self.lowlight_enhance]
    #     return self.detect_object(frame)

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
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 1)
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
    camera = Camera(fps_max,"./sample/home_night_01.mp4",True, detector=MobileDetDetector())
    # camera.set_marker_lines(VIDEO_05_MARKERLINES)
    camera.set_detect_box(0.25,0.05,0.65,0.75)
    camera.set_frame_skip(0)
    camera.set_selected_classes(["motorbike","person"])
    camera.set_preprocess_functions([camera.lowlight_enhance])
    camera.set_video_output("./output/home_night_02_mobileDet_lle.mp4")
    # camera.run()
    video_source = cv2.VideoCapture("./sample/home_night_02.mp4")
    i = 0
    inference_time_history = []
    ret, frame = video_source.read()
    while ret is not None and i < 500:
        ret, frame = video_source.read()
        if frame is None:
            continue           
        i+=1
        start = time.time()
        print("camera.py executing " + str(i))
        frame = camera.detect_object(frame)
        inference_time_history.append(time.time() - start)
        if len(inference_time_history) > 10:
            inference_time_history = inference_time_history[1:]

        if i%100 == 0 and i > 0:
            print("camera.py executing " + str(i))
            print("FPS: ", str(1/(sum(inference_time_history)/len(inference_time_history))))

        camera.write_frame_to_output_file(frame)

    camera.out.release()
    

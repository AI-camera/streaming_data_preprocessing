import cv2
import threading
from multiprocessing import Process
import time
import logging

logger = logging.getLogger(__name__)

thread = None
process = None

class Camera:
	def __init__(self,fps=20,video_source=0):
		logger.info(f"Initializing camera class with {fps} fps and video_source={video_source}")
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
		self.detector = None
		self.start_time = time.time()
		self.end_tine = time.time()

	def run(self):
		logging.debug("Perparing thread")
		global thread

		if thread is None:
			logging.debug("Creating thread")
			thread = threading.Thread(target=self._capture_loop,daemon=True)
			logger.debug("Starting thread")
			self.isrunning = True
			thread.start()
			logger.info("Thread started")
	# def run(self):
	# 	process = Process(target=self._capture_loop)
	# 	self.isrunning = True
	# 	process.start()

	def _capture_loop(self):
		# dt = 1/self.fps
		# logger.debug("Observation started")

		v, img = self.camera.read()
		# self.detector = OpticalFlowMotionDetector(img)

		while self.isrunning:
			v,img = self.camera.read()
			if v:
				# if len(self.frames)==self.max_frames:
				# 	self.frames = self.frames[1:]
				self.frames.append(img)
			# time.sleep(dt)
			# self.regulate_stream_fps()
		logger.info("Thread stopped successfully")

	# def stop(self):
	# 	logger.debug("Stopping thread")
	# 	self.isrunning = False

	def attach_fps(self, frame):
		return cv2.putText(frame, 'FPS: ' + str(self.stream_fps), (10,450), cv2.FONT_HERSHEY_SIMPLEX, 
						1,(0, 255, 0), 2, cv2.LINE_AA)
	
	# def encode_to_png(self,frame):
	# 	return cv2.imencode('.png',frame)[1].tobytes()
	
	def encode_to_jpg(self,frame):
		return cv2.imencode('.jpg',frame)[1].tobytes()

	def get_frame(self, _bytes=True):
		if len(self.frames)>0:
			frame_with_fps = self.attach_fps(self.get_frame_raw())
			if _bytes:
				img = self.encode_to_jpg(frame_with_fps)
			else:
				img = frame_with_fps
		else:
			with open("images/not_found.jpeg","rb") as f:
				img = f.read()
		return img
	
	def get_frame_raw(self):
		if len(self.frames)>0:
			return self.frames[-1]
	 
	# def regulate_stream_fps(self):
	# 	locktime = 2
	# 	try:
	# 		if(self.fps_lock and (time.time() - self.start_time) > locktime):
	# 			self.fps_lock = False
	# 		frame_raw = self.get_frame_raw()
	# 		fps_adjustment = np.ceil(self.detector.detect(frame_raw) % self.fps)
	# 		if(not self.fps_lock):
	# 			self.stream_fps = fps_adjustment

	# 		# Bump the fps up if there's motion
	# 		if(fps_adjustment > 3):
	# 			self.stream_fps = self.fps
	# 			self.fps_lock = True
	# 			self.start_time = time.time()
	# 	except Exception as e:
	# 		print(e)
	
	# def get_regulated_stream_fps(self):
	# 	return self.stream_fps

	
	def get_video(self):
		raw_frame = self.get_frame_raw()
		return self.encode_to_jpg(raw_frame)

	def get_detect_face_video(self):
		raw_frame = self.get_frame_raw()
		return self.encode_to_jpg(self.get_detect_face_frame(raw_frame))
import cv2

# import os
# os.environ['DISPLAY'] = ':0'

class VIQS:
    def __init__(self):
        pass

    def non_max_suppress(self,bboxes):
        pass

    #Compress a frame window
    def compress(self,frame_window):
        scale_percent = 50 # percent of original size
        original_width = frame_window.shape[1]
        original_height = frame_window.shape[0] 
        original_dim = (original_width, original_height)

        width = int(frame_window.shape[1] * scale_percent / 100)
        height = int(frame_window.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(frame_window, dim, interpolation = cv2.INTER_AREA)

        #Upscale?
        upscaled = cv2.resize(resized, original_dim, interpolation = cv2.INTER_AREA)
        return upscaled

    #bboxes[0] = (x1,y1,x2,y2)
    def viqs(self, frame, bboxes):
        bboxes = self.non_max_suppress(bboxes)
        


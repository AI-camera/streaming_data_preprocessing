import multiprocessing
from camera import Camera
from multiprocessing import Process
from multiprocessing import shared_memory

class CameraManager:
    def __init__(self):
        self.camera_dict = dict()
        self.camera_process_dict = dict()
    
    def add_camera(self,id,source,allow_loop=False,fps=24):
        """
        - id: id of camera in camera dictionary
        - source: source for camera to read frames from
        - allow_loop: Use this with video files to turn them into endless (looping) streams
        """

        self.create_camera(id,source,allow_loop,fps)

        # self.camera_process_dict[id] = Process(target=self.create_camera_process,args=[id,source,allow_loop,fps])
        # self.camera_process_dict[id].start()
        # self.camera_dict[id].join()
    
    def get_camera(self,id):
        return self.camera_dict[id]

    def get_all_ID(self):
        return self.camera_dict.keys()

    def get_all_camera(self):
        return self.camera_dict.values()

    def validate_ID(self,id):
        return id in self.camera_dict[id]

    def create_camera(self,id,source,allow_loop,fps):
        self.camera_dict[id] = Camera(fps,source,True)
        self.camera_dict[id].run()
        # print("proc 2: " + str(self.get_all_ID()))
    
    def shared_mem(self):
        pass
    
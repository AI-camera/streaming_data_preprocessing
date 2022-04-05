from track import Track
from utils import *

class TrackManager():
    def __init__(self):
        self.tracks = []
        self.markerline_cross_count = dict()
    
    def HandleNewTrack(self, trackPosition, trackID, markerlineDict, redlight_markerline_id):
        '''
        * Handle new track and old track's updates
        * Track position should be normalized before calling this function
        '''
        if trackID not in self.GetIDList():
            newTrack = Track(trackPosition, trackID)
            self.tracks.append(newTrack)
            newTrack.Activate()
        else:
            self.GetTrackByID(trackID).Activate()
            if self.GetTrackByID(trackID).Update(trackPosition):
                self.CheckMarkerLineCross(self.GetTrackByID(trackID), markerlineDict, redlight_markerline_id)

    def HandleNewTracks(self, tracked_boxes_and_ids, markerlineDict, redlight_markerline_id):
        '''
        * Handle multiple tracks' updates
        * Track position should be normalized before calling this function
        '''
        for track in self.tracks:
            track.Deactivate()
        if len(tracked_boxes_and_ids) > 0:
            track_boxes = [row[0:4] for row in tracked_boxes_and_ids]
            ids = [row[4] for row in tracked_boxes_and_ids]
            for track_box,id in zip(track_boxes,ids):
                self.HandleNewTrack(track_box, id, markerlineDict,redlight_markerline_id)

    def CheckMarkerLineCross(self,track:Track,markerlineDict, redlight_markerline_id=[]):
        for markerline in markerlineDict.items():
            markerlineID = markerline[0]
            try:
                markerlinePoint1 = markerline[1][0]
                markerlinePoint2 = markerline[1][1]
            except:
                print("tracker_manager.py - Markerline should be in format ((x1,y1),(x2,y2))")
            
            trackPoint1, trackPoint2 = track.GetCurrentRoute()
            if intersect(trackPoint1,trackPoint2,markerlinePoint1,markerlinePoint2):
                if (len(track.crossedMarkerlineIDs) == 0) and markerlineID in redlight_markerline_id:
                    track.crossedMarkerlineIDs.append('RED_CROSS')

                track.CrossMarkerline(markerlineID)
                if markerline in self.markerline_cross_count.keys():
                    self.markerline_cross_count[markerline] +=1
                else: 
                    self.markerline_cross_count[markerline] = 1

    def GetIDList(self)->int:
        return [track.id for track in self.tracks]
    
    def GetTrackByID(self, id)->Track:
        for track in self.tracks:
            if track.id == id:
                return track
    

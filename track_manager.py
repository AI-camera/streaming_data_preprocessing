from track import Track

class TrackManager():
    def __init__(self):
        self.tracks = []
    
    def HandleNewTrack(self, trackPosition, trackID):
        if trackID not in self.GetIDList():
            newTrack = Track(trackPosition, trackID)
            self.tracks.append(newTrack)
            newTrack.Activate()
        else:
            self.GetTrackByID(trackID).Update(trackPosition)
            self.GetTrackByID(trackID).Activate()

        
              
    def HandleNewTracks(self, tracked_boxes_and_ids):
        for track in self.tracks:
            track.Deactivate()
        if len(tracked_boxes_and_ids) > 0:
            track_boxes = [row[0:4] for row in tracked_boxes_and_ids]
            ids = [row[4] for row in tracked_boxes_and_ids]
            for track_box,id in zip(track_boxes,ids):
                self.HandleNewTrack(track_box, id)

    def GetIDList(self)->int:
        return [track.id for track in self.tracks]
    
    def GetTrackByID(self, id)->Track:
        for track in self.tracks:
            if track.id == id:
                return track
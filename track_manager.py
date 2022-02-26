from track import Track

class TrackManager():
    def __init__(self):
        self.tracks = []
    
    def HandleNewTrack(self, trackPosition, trackID):
        if trackID not in self.GetIDList():
            newTrack = Track(trackPosition, trackID)
            self.tracks.append(newTrack)
        else:
            self.GetTrackByID(trackID).Update(trackPosition)

    def GetIDList(self)->int:
        return [track.id for track in self.tracks]
    
    def GetTrackByID(self, id)->Track:
        for track in self.tracks:
            if track.id == id:
                return track
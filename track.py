from utils import block_distance

class Track:
    def __init__(self, position, id):
        '''
            - Vehicle track with history of positions
            - Args:
                * Position: (x1,y1,x2,y2)
                * id: id of track from SORT tracker
        '''
        self.id = id
        self.direction = None
        if len(position) is not 4:
            print("position must be in (x1,y1,x2,y2) format")
            return
        x1,y1,x2,y2 = position
        self.history = [(int((x1+x2)/2),int((y1+y2)/2))]
        self.isActive = False
        self.crossedMarkerlineIDs = []

    def Update(self, newPosition):
        ''' Return true if an update was made, false if not'''
        x1,y1,x2,y2 = newPosition
        newPosition = (int((x1+x2)/2),int((y1+y2)/2))
        # prevent the track from jiggling
        variance = block_distance(newPosition, self.GetCurrentPosition())
        if len(self.history) >= 1500:
            self.history = self.history[1:]
        
        self.history.append((int((x1+x2)/2),int((y1+y2)/2)))
        return True
        # else:
        #     return False

    def GetCurrentPosition(self):
        return self.history[len(self.history)-1]
    
    def GetCurrentRoute(self):
        '''Return the two last position of track'''
        return self.history[len(self.history)-1], self.history[len(self.history)-2]
    
    def Activate(self):
        self.isActive = True
    
    def Deactivate(self):
        self.isActive = False

    def CrossMarkerline(self, markerlineName):
        if markerlineName not in self.crossedMarkerlineIDs:
            self.crossedMarkerlineIDs.append(markerlineName)



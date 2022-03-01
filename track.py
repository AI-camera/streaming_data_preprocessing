
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

    def Update(self, newPosition):
        x1,y1,x2,y2 = newPosition
        self.history.append((int((x1+x2)/2),int((y1+y2)/2)))
        if len(self.history) >= 1500:
            self.history = self.history[1:]

    def GetCurrentPosition(self):
        return self.history[len(self.history)-1]
    
    def Activate(self):
        self.isActive = True
    
    def Deactivate(self):
        self.isActive = False


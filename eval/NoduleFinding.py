class NoduleFinding(object):
    '''
    Represents a nodule
    '''

    def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, diameterX=None,
                 diameterY = None, diameterZ = None,coordType="World",
                 CADprobability=None, noduleType=None,  seriesInstanceUID=None):
        # set the variables and convert them to the correct type
        self.id = noduleid
        self.coordX = coordX
        self.coordY = coordY
        self.coordZ = coordZ
        self.diameterX = diameterX
        self.diameterY = diameterY
        self.diameterZ = diameterZ
        self.coordType = coordType
        self.CADprobability = CADprobability
        self.noduleType = noduleType
        self.candidateID = None
        self.seriesuid = seriesInstanceUID
import cv2 as cv
from Localization import *
from CSVDTS import *
from CNNFirstPassFilter import TrainFPSC
"""
Create a training dataset for the first pass sign classifier
"""
FPSC_Data = DataSet("FPCData", "FPCDataset",False)
makeNewSet = False
if(makeNewSet):




    GVSRBPath = "/users/labnet/st11/jaxonm/Downloads/archive/Test"

    NoSignSet = LibraryLocalization("Tests/ForestTest.png", False, False)
    #making a dataset of n number of roadsigns for the classification road signs
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n], (64,64)), 0)

TrainFPSC(FPSC_Data)

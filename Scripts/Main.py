import cv2 as cv
from Localization import *
from CSVDTS import *
from CNNFirstPassFilter import TrainFPSC, imgProcess
"""
Create a training dataset for the first pass sign classifier
"""
FPSC_Data = DataSet("D:\\School\\Comp4301Project\\FPCData", "FPCDataset", False)

makeNewSet = False
trainNewModel = False

if(makeNewSet):

    GVSRBPath = "D:\\School\\Comp4301Project\\GTSRBSet\\Test"

    NoSignSet = LibraryLocalization("Tests\\ForestTest.jpg", True, False)
    #making a dataset of n number of roadsigns for the classification road signs
    setSize = 0
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n + setSize))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n][0], (64,64)), 0)
        setSize += 1

    NoSignSet = LibraryLocalization("Tests\\Houses_Test.jpg", True, False)
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n + setSize))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n][0], (64,64)), 0)
        setSize += 1
    NoSignSet = LibraryLocalization("Tests\\Houses_Test_2.jpg", True, False)
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n + setSize))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n][0], (64,64)), 0)
        setSize += 1
    NoSignSet = LibraryLocalization("Tests\\Houses_Test_3.jpg", True, False)
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n + setSize))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n][0], (64,64)), 0)
        setSize += 1
    NoSignSet = LibraryLocalization("Tests\\Houses_Test_4.jpg", True, False)
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n + setSize))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n][0], (64,64)), 0)
        setSize += 1
    NoSignSet = LibraryLocalization("Tests\\Desert_Test.jpg", True, False)
    for n in range(len(NoSignSet)):
        imagepath = os.path.join(GVSRBPath, "{:05d}.png".format(n + setSize))
        newItem = cv.resize(cv.imread(imagepath), (64,64))
        FPSC_Data.addItem(newItem, 1)
        FPSC_Data.addItem(cv.resize(NoSignSet[n][0], (64,64)), 0)
        setSize += 1


if(trainNewModel):
    TrainedModel = TrainFPSC(FPSC_Data)

imgProcess("Tests\\Road_Test.jpg")


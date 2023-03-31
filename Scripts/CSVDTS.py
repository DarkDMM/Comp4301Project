#CSV Dataset management functions for writing csv logged datset files
import os
import shutil
import csv
import cv2 as cv
import pandas as pd
import numpy as np

class DataSet:



    def __init__(self, data, labels, path, name, mode = "image"):
        """
    #creates a new dataset at the given path storing each item in data with a given label and loading a csv with the matches

    #arg data: the set of data to be saved
    #arg Labels: the class labels for each data item
    #arg Path: the bath to store the new dataset
    #arg Name: the name of the dataset
    """
        if (not self.dataValidation(data, labels)):
            return False
    
        if (not self.dataValidation(data, labels)):
            return False

        pathProposal = os.path.join(path, name)

        #check propsed path and ask for overwrite verification
        if(os.path.exists(pathProposal)):

            user_input = input("The specified file already exists, overwrite? (y/n): ").upper()

            #user decides to overwrite, shutil rms the diretory and process continues
            if(user_input == "Y"):
                shutil.rmtree(pathProposal)

            #user decides not to overwrite, process is exited
            elif(user_input == "N"):
                return False

        else:
        #make the general structure of the dataset
            os.mkdir(pathProposal)
            os.mkdir(os.path.join(pathProposal, "Data"))
            self.DataPath = os.path.join(pathProposal, "Data")
        #the defualt dataset creation mode is going to be for images
        if(mode == 'image'):

            ColumnLabels = ["Width", "Height", "Label", "Path"]

            manifestPath = os.path.join(pathProposal, (name + "_Manifest.csv"))
            self.ManifestPath = os.path.join(pathProposal, (name + "_Manifest.csv"))
            csvLog = open(manifestPath, 'a') 
            dicWriter = csv.DictWriter(csvLog, ColumnLabels)

            for i in range(len(data)):

                dataPath = os.path.join(pathProposal, "Data")
                #create the path to the image
                imgPath = dataPath + "/SetItem_" + str(i) + ".jpg"
                #get the dimensions of the image
                imWidth = data.shape[0]
                imHeight = data.shape[1]
                #get the label of the image
                imLabel = labels[i]

                ManiitemfestDict = {"Width":imWidth, "Height": imHeight,"Label":imLabel,"Path":imgPath}

                #write the image to the dataset
                cv.imwrite(imgPath, data[i])
                #write the entry to thedataset
                dicWriter.writerow(ManifestDict)

        self.size = len(data)
        csvLog.close()
    
    def __init__(self, path, name, autoOverwrite = None):
        pathProposal = os.path.join(path, name)
        #check propsed path and ask for overwrite verification
        if(os.path.exists(pathProposal)):
            if(autoOverwrite == None):
                user_input = input("The specified file already exists, overwrite? (y/n): ").upper()
            elif(autoOverwrite):
                user_input = 'Y'
            elif(not autoOverwrite):
                user_input = 'N'

            #user decides to overwrite, shutil rms the diretory and process continues
            if(user_input == "Y"):
                shutil.rmtree(pathProposal)
                os.mkdir(pathProposal)
                os.mkdir(os.path.join(pathProposal, "Data"))
                self.DataPath = os.path.join(pathProposal, "Data")
                self.ManifestPath = os.path.join(pathProposal, (name + "_Manifest.csv"))
                self.size = 0

            #user decides not to overwrite, process is exited
            elif(user_input == "N"):

                self.DataPath = os.path.join(pathProposal, "Data")
                self.ManifestPath = os.path.join(pathProposal, (name + "_Manifest.csv"))
                self.size = len((pd.read_csv(self.ManifestPath)).values)
        
        

    def extend(self, data, labels):
        """
        Extends the dataset with the provided data and labels
        """
        currentManifest = pd.read_csv(self.ManifestPath).data
        print()

    def addItem(self, item, label):
        ColumnLabels = ["Width", "Height", "Label", "Path"]
        csvLog = open(self.ManifestPath,"a")
        dicWriter = csv.DictWriter(csvLog, ColumnLabels)
        imgPath = self.DataPath + "/SetItem_" + str(self.size + 1) + ".jpg"
        #get the dimensions of the image
        imWidth = item.shape[0]
        imHeight = item.shape[1]
        #get the label of the image
        imLabel = label

        ManifestDict = {"Width":imWidth, "Height": imHeight,"Label":imLabel,"Path":imgPath}
        self.ManifestHeaders = ManifestDict.keys
        #write the image to the dataset
        cv.imwrite(imgPath, item)
        #write the entry to thedataset
        dicWriter.writerow(ManifestDict)
        self.size += 1


    def dataValidation(self, data, labels):
        if len(data != len(labels)):
            print("Dataset Creation Failed, dataset and label lengths differ")
            return False
        
    def loadData(self, batchSize = -1):
        if batchSize == -1:
            batchSize - self.size
        
        #create a list to store the output
        DataItems = []
        
        itemPaths = self.getManifest().drop("Width", 1).drop("Height", 1)
        for item in itemPaths.values:
            DataItems.append((cv.imread(str(item[1])) , item[0]))
        
        npItems = np.array(DataItems, dtype = 'object')
        np.random.shuffle(npItems)
        Cutoff = self.size // 10
        Test = npItems[0:Cutoff]
        Train = npItems[Cutoff:]

        return Train[0:, 0], Train[0:,1], Test[0:, 0], Test[0:,1]
            


    #returns the manifest of the data for this dataset as a pandas data set  
    def getManifest(self):
        return pd.read_csv(self.ManifestPath, names = ["Width", "Height", "Label", "Path"])


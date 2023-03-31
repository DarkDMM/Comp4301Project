#example object localization taken from https://www.learnopenCV.com/selective-search-for-object-detection-cpp-python
#standard library implementation of object localization

import sys
import cv2

"""runs the standard OpenCV library object selective search algorithm
   input image: the image to run segmentation on
   fastMode: sets the speed of the segmentation algorithm, fastmode is low quality but high speed
   Debug: whether console prints are engaged for debugging
"""




def LibraryLocalization(inputImage, fastMode, Debug):

    #not implementing the argMain and function argument checks as this is called from a function instead
    #optimize the Cv2 system for multithreading
    if(Debug):
        print("Segmenting Image:", inputImage)

    #cv2.setUseOptimized(True)
    #cv2.SetNumThreads(4)


    #read the input image
    im = cv2.imread(inputImage)
    #rezise the image
    newHeight = 200
    newWidth = int(im.shape[1]*200/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))

    #create Selective Search Segmentation object using defualt parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    #set input image for the segmentation
    ss.setBaseImage(im)

    #switch to fast but low recall selective search method
    if(fastMode):
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    #get the region proposals
    rects = ss.process()

    if(Debug):
        print("total number of region proposals: {}".format(len(rects)))

        numShowRects = 100
        increment = 50

        while True:
            #copy the original image
            ImageWithBoxes = im.copy()

            #iterate over proposed reagions
            for i, rect in enumerate(rects):
                if(i < numShowRects):
                    x,y,w,h = rect
                    cv2.rectangle(ImageWithBoxes, (x,y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    break
            #show output
            cv2.imshow("Output", ImageWithBoxes)
            #get key presses
            k = cv2.waitKey(0) & 0xff

            #m is pressed
            if(k == 109):
                #increase total unmber of recatngles by increment
                numShowRects += increment
            #l is pressed
            if(k == 108 and numShowRects > increment):
                numShowRects -= increment
            #q is pressed
            elif k == 113:
                break
        cv2.destroyAllWindows()

    #foir the purposes of ciommit
    SegmentedSet = []
    #crop the input imag for each proposed ROI
    for i, rect in enumerate(rects):
        ImgSegment = im.copy()
        x,y,w,h = rect
        ImgSegment = ImgSegment[y:y+h, x:x+w]
        SegmentedSet.append(ImgSegment)




    print("Image segmentation complete")
    return SegmentedSet



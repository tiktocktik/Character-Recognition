import cv2
import numpy as np
from WordSegmentation import wordSegmentation, prepareImg
import os
import sys
MAIN_FOLDER = os.getcwd()

def word_Segmentation(path,inputFileName):
    newFile = inputFileName.split('.')
    img = prepareImg(cv2.imread(path),60)
    img = cv2.resize(img,(0,0),fx=2.5,fy=2.5)
    wordRes = wordSegmentation(img, kernelSize=27, sigma=13, theta=7, minArea=250)
    if not os.path.exists(MAIN_FOLDER+"/processed_images\\%s"%newFile[0]):
        os.makedirs(MAIN_FOLDER+"/processed_images\\%s"%newFile[0])
    temp = MAIN_FOLDER+"/processed_images\\%s"%newFile[0]
    for(j,w) in enumerate(wordRes):
        (wordBox,wordImg) = w
        (x,y,w,h) = wordBox
        cv2.imwrite(MAIN_FOLDER+"/processed_images/%s/%d.png"%(newFile[0], j), wordImg)
        cv2.rectangle(img,(x,y),(x+w,y+h),0,1)
        cv2.imshow('result',img)
        cv2.waitKey(0)
    return temp

def character_Segmentation(path,inputFileName):
    charFiles = os.listdir(path)
    print('charfiles = ',charFiles)
    for(i,f) in enumerate(charFiles):
            newFile =f.split('.')
            img = prepareImg(cv2.imread(path+'\\%s'%(f)),210)

            if not(os.path.exists(MAIN_FOLDER+"/processed_character_images/%s/%s"%(inputFileName,i))):
                os.makedirs(MAIN_FOLDER+"/processed_character_images/%s/%s"%(inputFileName,i))

            charRes = wordSegmentation(img, kernelSize=25, sigma=1, theta=1, minArea=4000)

            for(j,w) in enumerate(charRes): 
                (charBox, charImg) = w
                (x,y,w,h) = charBox
                cv2.imwrite(MAIN_FOLDER+"/processed_character_images/%s/%s/%d.png"%(inputFileName,i, j), charImg)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow('result',img)
                cv2.waitKey(0)
    returnPath = MAIN_FOLDER+'/processed_character_images/%s/'%(inputFileName)
    return returnPath

def singleCharacterSegmentation(path,inputFileName):
    newCharFile = inputFileName.split('.')
    print(path , inputFileName)
    # sys.exit()
    img = prepareImg(cv2.imread(path),210)
    img = cv2.resize(img,(0,0),fx=2.5,fy=2.5)
    charRes = wordSegmentation(img, kernelSize=101, sigma=5, theta=2, minArea=4100)
    if not(os.path.exists(MAIN_FOLDER+"/processed_single_character_images/%s"%(newCharFile[0]))):
        os.makedirs(MAIN_FOLDER+"/processed_single_character_images/%s"%(newCharFile[0]))
    for(j,w) in enumerate(charRes):
        print(j)
        (charBox, charImg) = w
        (x,y,w,h) = charBox
        cv2.imwrite(MAIN_FOLDER+"/processed_single_character_images/%s/%d.png"%(newCharFile[0], j), charImg)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('result',img)
        cv2.waitKey(0)
    returnPath = MAIN_FOLDER+'/processed_single_character_images/%s'%(newCharFile[0])
    return returnPath

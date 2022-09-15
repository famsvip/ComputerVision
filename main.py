import os
import matplotlib
from Algorithm.main import *
import cv2
import argparse
import time
import requests
import yaml
import tqdm
import torchvision
from turtle import color
import pandas as pd
import seaborn
import numpy as np

# M1 SSL Fix to load 
# torch model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

matplotlib.use('TKAgg')
# Disable tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def det_move(obj_x_coord, obj_y_coord, xres, yres):
    centerx, centery = xres/2, yres/2

    move_x = obj_x_coord-centerx
    move_y = obj_y_coord-centery
    if(move_x != 0):
        move_x /= abs(obj_x_coord-centerx)
    if(move_y != 0):
        move_y /= abs(obj_y_coord-centery)
    return(move_x, move_y)


def main(_argv):

    #parser = argparse.ArgumentParser()
    # Initialize Camera Intel Realsense


    # Parse arguments
    # if _argv.Debug == "1" or _argv.D == "1":
    #    Debug_flag = 1
    #    # Create window for video
    #    cv2.namedWindow("Video")
    #    cv2.namedWindow("Video_Depth")

    # elif _argv.Debug == "0" or _argv.D == "0":
    #   Debug_flag = 0

    # Load saved CV model
    model = get_model()

    # Initialize Algorithm
    oldCords = None
    depth = None

    while True:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            # Make detections
            results = model(frame)

            #write coordinates
            #print('\n', results.xyxy[0]) #results have to be cleaned up

            #print('\n', results.pandas().xyxy[0]) #this works, but the vector may not be workable for movement team
            #print(type(results.pandas().xyxy[0]))


            dataframe = results.pandas().xyxy[0]
            xLeft = dataframe.iat[0, 0]
            xRight = dataframe.iat[0, 2]
            yLower = dataframe.iat[0, 1]
            yUpper = dataframe.iat[0, 3]
            confidence = dataframe.iat[0, 4]
            item = dataframe.iat[0, 6]
            xCoor = (xRight+xLeft) / 2
            yCoor = (yUpper+yLower) / 2

            print('\n', results.pandas().xyxy[0])
            # print('\n', xLeft, yLower, xRight, yUpper)
            print('\n', xCoor, yCoor, item, confidence)

            cv2.imshow('YOLO', np.squeeze(results.render()))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
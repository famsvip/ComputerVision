import torch
import torchvision.transforms as transforms
from cv2 import (VideoCapture, namedWindow, imshow, waitKey, destroyWindow, imwrite)
import cv2 as cv
import numpy as np
import cv2
import pandas as pd
import time
import imageio
from matplotlib import cm
import math
import tkinter
from tkinter import Frame
from matplotlib.font_manager import FontProperties
import cv2
import PIL
from PIL import ImageTk, Image, ImageFile
import tkinter as tk
import csv
import os

from datetime import datetime
import matplotlib.pyplot as plt
# from detect import run as yolorun
import threading
import dataframe_image as dfi
from tkinter import Tk, Button, PhotoImage
from tkinter import *
from ultralytics import YOLO
import urllib.request
from urllib.error import HTTPError
import subprocess
from IPython.display import display, Javascript

# %%
if os.path.exists("Curr_Tumor.png"):
    os.remove("Curr_Tumor.png")


# %%
class Graph_Two:
    def __init__(self, data):
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.patch.set_facecolor('#121212')
        self.ax.set_facecolor('#121212')
        self.ax.spines['bottom'].set_color('#fff')
        self.ax.spines['left'].set_color('#fff')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(axis='x', colors='#fff')
        self.ax.tick_params(axis='y', colors='#fff')
        self.ax.set_title("YOLOv5 Accuracy per Detection", color='#fff')
        self.ax.set_xlabel("Detection", color='#fff')
        self.ax.set_ylabel("YOLOv5 Accuracy ", color='#fff')
        self.data = data
        #         self.dv = dv
        self.draw()
        self.fig.savefig('acc_arr.png')

    def draw(self):
        x = np.arange(len(self.data))
        y = self.data
        #         y2 = self.dv
        self.ax.plot(x, y, color='#0000FF', linewidth=2)
        #         self.ax.plot(x, y2, color='#5d00ff',linewidth=2)
        max_val = np.max(y)
        min_val = np.min(y)
        self.ax.set_ylim(0, 1)
        self.fig.canvas.draw()


# %%
class Graph:
    def __init__(self, data):
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.patch.set_facecolor('#121212')
        self.ax.set_facecolor('#121212')
        self.ax.spines['bottom'].set_color('#fff')
        self.ax.spines['left'].set_color('#fff')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(axis='x', colors='#fff')
        self.ax.tick_params(axis='y', colors='#fff')
        self.ax.set_title("Relative Depth per Detection", color='#fff')
        self.ax.set_xlabel("Detection", color='#fff')
        self.ax.set_ylabel("Relative Depth", color='#fff')
        self.data = data
        #         self.dv = dv
        self.draw()
        self.fig.savefig('my_plot.png')

    def draw(self):
        x = np.arange(len(self.data))
        y = self.data
        #         y2 = self.dv
        self.ax.plot(x, y, color='#ff0066', linewidth=2)
        #         self.ax.plot(x, y2, color='#5d00ff',linewidth=2)
        max_val = np.max(y)
        min_val = np.min(y)
        self.ax.set_ylim(0, 1)
        self.fig.canvas.draw()


def create_table(x, y, z, depth, distance_total):
    df = pd.DataFrame({'X': [str(round(x, 2))],
                       'Y': [str(round(y, 2))],
                       'Z': [str(round(z, 2))],
                       'DEPTH': [str(round(depth, 2))],
                       'DISTANCE': [str(round(distance_total, 2))]}).style.hide_index().set_table_styles(
        [{'selector': 'th', 'props': [
            ('background-color', 'black'),
            ('color', 'white'),
            ('border', 'none'),
        ], }, {'selector': 'tr', 'props': [('border-top', '1px solid white'),
                                           ('border-bottom', '1px solid white')]}]).set_properties(
        **{'background-color': 'black', 'color': 'white'})

    dfi.export(df, 'dataframe.png')

    image = PIL.Image.open('dataframe.png')


# %%
def function_one(acc_level):
    weight_type = "best_3.pt"
    #    source_type = "test_video.mp4"
    #    source_type = "SubmissionRecording.mp4"
    source_type = "0"
    conf = "0.75"

    command = f"python3 /Users/alifakhry/Downloads/yolov5-master/detect.py --weights {weight_type} --conf {conf} --source {source_type} --var-acc {acc_level}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


# %%
import time


def function_two(acc_level):
    adjust = root_h // 9

    total_data_avg = []
    gyro_arr = []
    last_length = len(total_data_avg)

    with open("test1.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            gyro_arr.append(row)

    indx = 0

    graph = Graph([0])
    create_table(0, 0, 0, 0, 0)

    im = PIL.Image.open("my_plot.png")
    im = im.resize((int(ratio_w * 462), int(ratio_h * 320)))
    tk_image_2 = ImageTk.PhotoImage(im)
    image_label_2.configure(image=tk_image_2)
    image_label_2.image = tk_image_2
    image_label_2.place(x=root_w // 2 + root_w // 8, y=root_h // 12 + adjust)

    #     im = PIL.Image.open("dataframe.png")
    #     im = im.resize((int(ratio_w * 462), int(ratio_h * 120)))
    #     tk_image_3 = ImageTk.PhotoImage(im)
    #     image_label_3.configure(image=tk_image_3)
    #     image_label_3.image = tk_image_3
    #     image_label_3.place(x=root_w//2 + root_w//8, y=root_h//2)

    im = None

    #     if acc_level == 1:
    #                         im = PIL.Image.open("button_high-speed-camera-mode.png")
    #     elif acc_level == 2:
    #                         im = PIL.Image.open("button_balanced-camera-mode.png")
    #     elif acc_level == 3:
    #                         im = PIL.Image.open("button_high-accuracy-camera-mode.png")

    #     im = im.resize((int(root_w//2), int(ratio_h * 81)))
    #     tk_image_4 = ImageTk.PhotoImage(im)
    #     image_label_4.configure(image=tk_image_4)
    #     image_label_4.image = tk_image_4
    #     image_label_4.place(x=root_w//20, y=root_h//2 + root_h//5)

    while (True):

        try:

            im = PIL.Image.open("Curr_Tumor.png")
            im = im.resize((int(root_w // 2), int(root_h // (9 / 5))))
            tk_image = ImageTk.PhotoImage(im)
            image_label.configure(image=tk_image)
            image_label.image = tk_image
            image_label.place(x=root_w // 20, y=root_h // 12 + adjust)

        except:
            continue
        with open("data.csv", newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                total_data_avg.append(float(row[0]))

        if len(total_data_avg) > 0 and len(gyro_arr) > 0:

            if indx >= len(gyro_arr):
                indx = 0

            gyro_new_data = gyro_arr[indx]

            if len(total_data_avg) > 1 and len(total_data_avg) > last_length:
                #                     create_table(float(gyro_new_data[0]),float(gyro_new_data[1]), float(gyro_new_data[2]),float(total_data_avg[-1]),float(gyro_new_data[0]))
                last_length = len(total_data_avg)
                Graph(total_data_avg)
                try:
                    im = PIL.Image.open("my_plot.png")
                    im = im.resize((int(ratio_w * 462), int(ratio_h * 320)))
                    tk_image_2 = ImageTk.PhotoImage(im)
                    image_label_2.configure(image=tk_image_2)
                    image_label_2.image = tk_image_2
                    image_label_2.place(x=root_w // 2 + root_w // 8, y=root_h // 12 + adjust)

                #                         im = PIL.Image.open("dataframe.png")
                #                         im = im.resize((int(ratio_w * 462), int(ratio_h * 120)))
                #                         tk_image_3 = ImageTk.PhotoImage(im)
                #                         image_label_3.configure(image=tk_image_3)
                #                         image_label_3.image = tk_image_3
                #                         image_label_3.place(x=root_w//2 + root_w//8, y=root_h//2)

                except:
                    continue
            elif len(total_data_avg) > last_length:
                create_table(float(gyro_new_data[0]), float(gyro_new_data[1]), float(gyro_new_data[2]),
                             float(total_data_avg[-1]), float(gyro_new_data[0]))
            #                 try:
            #                         last_length = len(total_data_avg)
            #                         im = PIL.Image.open("dataframe.png")
            #                         im = im.resize((int(ratio_w * 462), int(ratio_h * 120)))
            #                         tk_image_3 = ImageTk.PhotoImage(im)
            #                         image_label_3.configure(image=tk_image_3)
            #                         image_label_3.image = tk_image_3
            #                         image_label_3.place(x=root_w//2 + root_w//8, y=root_h//2)
            #                 except:
            #                     continue
            #             print(total_data_avg)
            #             print(gyro_new_data)
            total_data_avg = []
        root.update()
        indx += 1
    print("DONE")

    # %%
    from screeninfo import get_monitors
    frame = None
    root_w, root_h = 0, 0
    ratio_w, ratio_h = 0, 0
    for m in get_monitors():
        root_w = (int(str(str(m).split(",")[2])[7:]))
        root_h = int((int(str(str(m).split(",")[3])[8:])) - 100)
        break
    mybutton = None
    ratio_w = root_w / 1440
    ratio_h = root_h / 800
    Running_True = [None]
    Running_True[0] = True
    root = tk.Toplevel()
    root.geometry(f"{str(root_w)}x{str(int(root_h))}")
    root.configure(bg="#161618")
    root.title('Depth GUI')
    root.curr = 0
    root.primary = True
    frame_curr = Frame(root)
    frame_curr.pack()
    image_label_4 = tk.Label(root)
    root.buttons = []
    root.main_button = []
    root.running_prim = True
    adjust = root_h // 9

    #     def setting_choice():
    #             mybutton_new_1 = tk.Button(root, image=low_acc, command = lambda: change_back(1))
    #             mybutton_new_1.place(x=((root_w - low_acc.width()) // 2), y=((root_h - low_acc.height()) // 4))
    #             mybutton_new_2 = tk.Button(root, image=med_acc, command = lambda: change_back(2))
    #             mybutton_new_2.place(x=((root_w - low_acc.width()) // 2), y=2 *((root_h - med_acc.height()) // 4))
    #             mybutton_new_3 = tk.Button(root, image=high_acc, command = lambda: change_back(3))
    #             mybutton_new_3.place(x=((root_w - low_acc.width()) // 2), y=3 * ((root_h - high_acc.height()) // 4))
    #             root.buttons = [mybutton_new_1,mybutton_new_2,mybutton_new_3]
    def change_back(acc_level):
        #             root.buttons[0].destroy()
        #             root.buttons[1].destroy()
        #             root.buttons[2].destroy()
        mybutton = tk.Button(root, image=pressed_image, command=update, highlightbackground="white",
                             highlightthickness=1)
        mybutton.place(x=root_w // 2 + root_w // 8, y=root_h // 2 + root_h // 26 + adjust)
        root.main_button = [mybutton]
        root.curr = 1
        Running_True[0] = True
        root.update()
        thread2 = threading.Thread(target=function_two, args=(1,))
        thread2.start()
        function_one(acc_level)

    def update():
        image_label_6.destroy()
        mybutton = root.main_button[0]
        if root.curr == 0:
            if root.primary:
                root.primary = False
                mybutton.destroy()
                change_back(1)
            else:
                mybutton.configure(image=pressed_image)
                root.curr = 1
                Running_True[0] = True
                update_image()
        else:
            javascript_code = "Jupyter.notebook.session.delete();"
            display(Javascript(javascript_code))
        root.update()

    im_1 = PIL.Image.open("button_start.png")
    im_1 = im_1.resize((int(ratio_w * 462), int(ratio_h * 81)))
    im_1.save("button_start.png")
    im_2 = PIL.Image.open("button_reset.png")
    im_2 = im_2.resize((int(ratio_w * 462), int(ratio_h * 81)))
    im_2.save("button_reset.png")
    im_3 = PIL.Image.open("button_high-speed-camera-mode.png")
    im_3 = im_3.resize((int(root_w / (70 / 23)), int(ratio_h * 81)))
    im_3.save("button_high-speed-camera-mode.png")
    im_4 = PIL.Image.open("button_balanced-camera-mode.png")
    im_4 = im_4.resize((int(root_w / (70 / 23)), int(ratio_h * 81)))
    im_4.save("button_balanced-camera-mode.png")
    im_5 = PIL.Image.open("button_high-accuracy-camera-mode.png")
    im_5 = im_5.resize((int(root_w / (70 / 23)), int(ratio_h * 81)))
    im_5.save("button_high-accuracy-camera-mode.png")
    im_6 = PIL.Image.open("button_start_blue.png")
    im_6 = im_6.resize((int(ratio_w * 462), int(ratio_h * 81)))
    im_6.save("button_start_blue.png")
    normal_image = PhotoImage(file="button_start.png")
    pressed_image = PhotoImage(file="button_reset.png")
    low_acc = PhotoImage(file="button_high-speed-camera-mode.png")
    med_acc = PhotoImage(file="button_balanced-camera-mode.png")
    high_acc = PhotoImage(file="button_high-accuracy-camera-mode.png")
    start_blue = PhotoImage(file="button_start_blue.png")
    image_label = tk.Label(root)
    image_label_2 = tk.Label(root)
    image_label_3 = tk.Label(root)
    image_label_4 = tk.Label(root)
    image_label_5 = tk.Label(root)
    image_label_6 = tk.Label(root)
    im = PIL.Image.open("TAJ_FRAME.png")
    im = im.resize((int(root_w), int(root_h + 100)))
    tk_image_6 = ImageTk.PhotoImage(im)
    image_label_6.configure(image=tk_image_6)
    image_label_6.image = tk_image_6
    image_label_6.place(x=0, y=0, relwidth=1, relheight=1)
    mybutton = tk.Button(root, image=start_blue, command=update, highlightbackground="white",
                         highlightthickness=2)
    mybutton.place(x=(root_w - normal_image.width()) // 2, y=int(2.3 * (root_h - normal_image.height()) // 3))
    root.main_button = [mybutton]
    padding = 50
    text_box_label = tk.Label(root)
    root.mainloop()


# %%
import numpy as np
import matplotlib.pyplot as plt


class Graph_Data:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(axis='x')
        self.ax.tick_params(axis='y')
        self.ax.set_title("Real Distance and Relative Depth (MiDaS Algorithms)")
        self.ax.set_xlabel("Real Distance (cm)")
        self.ax.set_ylabel("Detected Relative Depth")
        self.draw()
        self.fig.savefig('my_plot.png')

    def draw(self):
        x = np.linspace(1.5, 6.0, 100)
        x2 = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

        # Calculate and plot the cubic equations
        y1 = 0.0116 * x ** 3 - 0.0675 * x ** 2 + 0.0120 * x + 0.7234
        y2 = -0.0159 * x ** 3 + 0.1867 * x ** 2 - 0.5623 * x + 0.8235
        y3 = -0.0353 * x ** 3 + 0.3498 * x ** 2 - 0.9193 * x + 1.0145

        y4 = np.array([0.702, 0.424, 0.639, 0.271, 0.549, 0.451, 0.624, 0.443, 0.498, 0.973])
        y5 = np.array([0.325, 0.275, 0.427, 0.549, 0.286, 0.431, 0.592, 0.812, 0.804, 0.651])
        y6 = np.array([0.314, 0.267, 0.188, 0.859, 0.541, 0.251, 0.847, 0.929, 0.651, 0.427])

        self.ax.plot(x, y1, color='#ff0066', linewidth=2, label='DPT-Large')
        self.ax.plot(x, y2, color='green', linewidth=2, label='DPT-Hybrid')
        self.ax.plot(x, y3, color='#0066ff', linewidth=2, label='MiDaS-Small')

        self.ax.scatter(x2, y4, color='#ff0066', s=40)
        self.ax.scatter(x2, y5, color='green', s=40)
        self.ax.scatter(x2, y6, color='#0066ff', s=40)

        max_val = np.max(np.concatenate((y1, y2, y3)))
        min_val = np.min(np.concatenate((y1, y2, y3)))
        self.ax.set_ylim(0, 1)
        self.fig.canvas.draw()

        self.ax.legend()


Graph_Data()

# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1.5, 3.0, 100)

x_linear_1 = np.linspace(1.5, 3.0, 100)

x_linear_2 = np.linspace(3.0, 4.5, 100)

x_linear_3 = np.linspace(4.5, 6.0, 100)

eq1 = 0.0044 * x ** 3 - 0.0747 * x ** 2 + 0.4347 * x - 0.1995
linear_eq_1 = (0.168 * (x_linear_1 - 1.5)) + 0.299
# linear_eq_2 = (0.063 * (x_linear_2 - 3.0)) + 0.551
# linear_eq_3 = (0.017 * (x_linear_3 -4.5)) + 0.645

plt.figure(figsize=(10, 6))
plt.plot(x, eq1, label='Custom Algorithm')
plt.plot(x_linear_1, linear_eq_1, label='Linear Estimate', linestyle='--', color='black', linewidth=2)
# plt.plot(x_linear_2, linear_eq_2, linestyle='--', color='black', linewidth=2)
# plt.plot(x_linear_3, linear_eq_3, linestyle='--', color='black', linewidth=2)


plt.xlabel('True Distance (cm)')
plt.ylabel('Detected Relative Depth')
plt.legend()

plt.grid(True)
plt.title('Real Distance and Relative Depth')
plt.ylim(0.2, 0.6)
plt.show()
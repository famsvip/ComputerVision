import argparse
import os
import platform
import sys
from pathlib import Path
from matplotlib import cm
import imageio
import math
import csv
import torch
import time
import numpy as np
from PIL import Image


import cv2
import torch.backends.cudnn as cudnn

from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImagesAndLabels, LoadStreams
from yolov5_ros.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5_ros.utils.plots import Annotator, colors, save_one_box
from yolov5_ros.utils.torch_utils import select_device, time_sync

from yolov5_ros.utils.datasets import letterbox

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from . import blur_det_main_ as blur
def detect_blurry(img):
    # img = cv2.imread(image_path)
    img_fft, val, blurry = blur.blur_detector(img)
    print('Blurry' if blurry else 'Not Blurry')
    return val, blurry


def depth_estimate(x):
            if (x < 0.1):
                x = 1 - x
                return 1/(1 + math.e ** (-30 * (x-1))) + depth_estimate(0.1)
            elif (x < 0.3):
                x = 1 - x
                return 1/(1 + math.e ** (-15 * (x-1))) + depth_estimate(0.3)
            else:
                x = 1 - x
                return 1/(1 + math.e ** (-10 * (x-1)))

class yolov5_demo():
    def __init__(self, weights,
                        source,
                        data,
                        imagez,
                        conf_thres,
                        iou_thres,
                        max_det,
                        device,
                        view_img,
                        save_txt,
                        save_conf,
                        save_crop,
                        nosave,
                        classes,
                        agnostic_nms,
                        augment,
                        visualize,
                        update,
                        project,
                        name,
                        exist_ok,
                        line_thickness,
                        hide_labels,
                        hide_conf,
                        half,
                        dnn,
                        vid_stride,
                        var_acc
                        ):
        
        self.weights=weights  # model path or triton URL
        self.source='0'  # file/dir/URL/glob/screen/0(webcam)
        self.data=data  # dataset.yaml path
        self.imagez=imagez  # inference size (height, width)
        self.conf_thres=conf_thres  # confidence threshold
        self.iou_thres=iou_thres  # NMS IOU threshold
        self.max_det=max_det  # maximum detections per image
        self.device=device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=view_img  # show results
        self.save_txt=save_txt  # save results to *.txt
        self.save_conf=save_conf  # save confidences in --save-txt labels
        self.save_crop=save_crop  # save cropped prediction boxes
        self.nosave=nosave  # do not save images/videos
        self.classes=classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=agnostic_nms  # class-agnostic NMS
        self.augment=augment  # augmented inference
        self.visualize=visualize  # visualize features
        self.update=update  # update all models
        self.project=project  # save results to project/name
        self.name=name  # save results to project/name
        self.exist_ok=exist_ok  # existing project/name ok, do not increment
        self.line_thickness=line_thickness  # bounding box thickness (pixels)
        self.hide_labels=hide_labels  # hide labels
        self.hide_conf=hide_conf  # hide confidences
        self.half=half  # use FP16 half-precision inference
        self.dnn=dnn  # use OpenCV DNN for ONNX inference
        self.vid_stride=vid_stride  # video frame-rate stride
        self.var_acc=var_acc #type of version

        self.s = str()

        self.load_model()

    def load_model(self):
        # Load model
        imgsz = self.imagez
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        
        # Unpack only the necessary values
        stride, self.names, pt, jit, onnx, engine = (
            self.model.stride, 
            self.model.names, 
            self.model.pt, 
            self.model.jit, 
            self.model.onnx, 
            self.model.engine
        )
        
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Directories
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Half precision (FP16)
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        source = 0
        # Dataloader
        webcam = True

        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True

        bs = 1 
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        # Warmup the model
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], [0.0, 0.0, 0.0]


    # callback ==========================================================================

    # return ---------------------------------------
    # 1. class (str)                                +
    # 2. confidence (float)                         +
    # 3. x_min, y_min, x_max, y_max (float)         +
    # ----------------------------------------------
    def image_callback(self, image_raw):
        class_list = []
        #confidence_list = []
        conf_arr = []
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []
        x_TOTAL_DETECT = 0
        start_time = time.time()
        model_type = None
        
        #if self.var_acc == 1:
        #            model_type = "MiDaS_small"
        #elif self.var_acc == 2:
        #            model_type = "DPT_Hybrid"
        #else:
        #            model_type = "DPT_Large"

        #midas = torch.hub.load("intel-isl/MiDaS", model_type)
        #self.device = select_device(self.device)
        #midas.to(self.device)
        #midas.eval()
        #midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        #transform = None
        #if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        #            transform = midas_transforms.dpt_transform
        #else:
        #            transform = midas_transforms.small_transform
        
        #source = str(self.source)
        #self.save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        #is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        #is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        #webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        #screenshot = source.lower().startswith('screen')

        #if is_url and is_file:
        #    source = check_file(source)  # download
        
        
        
        # im is  NDArray[_SCT@ascontiguousarray
        # im = im.transpose(2, 0, 1)
        self.stride = 32  # stride
        self.img_size = 640
        img = letterbox(image_raw, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1
        
        # Inference
        save_dir = "runs/detect/exp1"
        path = ['0']

        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            # print("det:", det)
            self.seen += 1
            im0s = image_raw
            self.s += f'{i}: '


            save_path = save_dir + "/Curr_Tumor.png"  # im.jpg
            #p = Path(save_path)  # to Path
            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #imc = im0s.copy() if self.save_crop else im0s  # for save_crop
            annotator = Annotator(im0s, line_width=self.line_thickness, example=str(self.names), pil=True)
        
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                cords = []
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                input_batch = None
                img = None
        
#                if not webcam:
#                    input_batch = transform(im0s).to(device)
#                    img = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)
#                else:
#                    input_batch = transform(im0s[i]).to(device)
#                    img = cv2.cvtColor(im0s[i], cv2.COLOR_BGR2RGB)
#                with torch.no_grad():
#                    prediction = midas(input_batch)
#
#                    prediction = torch.nn.functional.interpolate(
#                        prediction.unsqueeze(1),
#                        size=img.shape[:2],
#                        mode="bicubic",
#                        align_corners=False,
#                    ).squeeze()
#
#                output = prediction.cpu().numpy()
#
#                plt.im-("Test.png", output, cmap=cm.gray)
#                new_im = imageio.v2.imread("Test.png")
#                img_updated = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
#                img_reverted = cv2.bitwise_not(img_updated)
#                img_new = img_reverted / 255.0
                high_depth = 1.01
                t_conf = None
                curr_res = None
                for *xyxy, conf, cls in reversed(det):
                    if conf > self.conf_thres:
                        cords.append(xyxy)
                        conf_arr.append(conf)
                        
                cords = [[int(value) for value in row] for row in cords]
                if len(cords):
                    
                    w,h = 640, 480

                    #w, h = im0s[i].shape[1], im0s[i].shape[0]
                
                    x1_t_F, y1_t_F, x2_t_F, y2_t_F = None,None,None,None
                    for indx, cord in enumerate(cords):
                        x1_t, y1_t, x2_t, y2_t = cord[0]-1, cord[1]-1, cord[2]-1, cord[3]-1

                        if (x1_t >= 10 and x2_t < w-10 and y1_t >= 10 and y2_t < h-10):
                            curr_depth = (abs(y2_t - y1_t) * abs(x2_t - x1_t)) / (w * h)
                            curr_depth = float(format(depth_estimate(curr_depth), '.3f'))
                            if curr_depth < high_depth:
                                curr_res = cord
                                high_depth = curr_depth
                                t_conf = conf_arr[indx]
                                x1_t_F, y1_t_F, x2_t_F, y2_t_F = x1_t, y1_t, x2_t, y2_t
                    
                    
                    
                    
                    if x1_t_F != None:
                        x_TOTAL_DETECT  += 1
                        
                        x1_t, y1_t, x2_t, y2_t = x1_t_F, y1_t_F, x2_t_F, y2_t_F

                        dv = (abs(y2_t - y1_t) * abs(x2_t - x1_t)) / (w * h)
                        dv = float(format(depth_estimate(dv), '.3f'))
                        depth_avg = dv
                        c = int(cls)
                        diameter = (abs(x1_t - x2_t) + abs(y1_t - y2_t)) // 2
                        curr_frame = np.array(annotator.result())
                        curr_frame = curr_frame[max(y1_t-100, 0):min(y2_t+100, h), max(x1_t-100, 0):min(x2_t+100, w)]
                        #cv2.imshow("cropped", curr_frame)
                        # cv2.imwrite("cropped.jpg", cropped)
                        val, blurry = detect_blurry(curr_frame)
                        # label = None if hide_labels else (
                        # names[c] if hide_conf else f'Depth:{depth_avg} Conf:{t_conf:.2f} Blur:{blurriness}')

                        # blur_dict_point5 = {5.1: 52, 1.4: 65, -2.4:79, -5.2:93, -6.9:110, -8.7:130, -8.9:205}
                        function = (0.5, "56.2 + (-3.51)*val + 0.845*(val**2)")
                        reference_diameter = eval(function[1])
                        reference_size = function[0]
                        actual_size = reference_size * diameter / reference_diameter
                        low, high = math.floor(actual_size), math.ceil(actual_size)
                        # label = None if hide_labels else (
                        #    names[c] if hide_conf else f'Depth:{depth_avg} Diameter:{diameter} Blur:{val:.2f}
                        #    f'Size:{actual_size:.2f} ({low}cm-{high}cm)')
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else
                                                          f'Diameter:{diameter} Blur:{val:.2f} Size:{actual_size:.2f} ({low}cm-{high}cm)')
                        annotator.box_label(curr_res, label, color=(255, 0, 0))

                        print("width:", w, " height:", h)
                        radius = 50
                        curr_frame = np.array(annotator.result())
                        if x1_t < w//2 - radius < w//2 + radius < x2_t and y1_t < h//2 - radius < h//2 + radius < y2_t:
                            cv2.circle(curr_frame, (w // 2, h // 2), radius, (0, 255, 0), 2)
                        else:
                            cv2.circle(curr_frame, (w // 2, h // 2), radius, (0, 0, 255), 2)
                        
                        
                        #label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'Depth:{depth_avg} Conf:{t_conf:.2f} d:{diameter}')
                        #
                        #annotator.box_label(curr_res, label, color=(255, 255, 0))
                        #radius = 50
                        #curr_frame = np.array(annotator.result())
                        #if abs(diameter-2*radius) > 40:
                        #    cv2.circle(curr_frame, (w // 2, h // 2), radius, (0, 0, 255), 2)
                        #else:
                        #    cv2.circle(curr_frame, (w // 2, h // 2), radius, (0, 255, 0), 2)

                        #    annotator.box_label([(w // 2 - 30000 // w, h // 2 - 30000 // h), (w // 2 + 30000 // w, h // 2 + 30000 // h)],
                        #                        None, color=(0, 255, 0))

                        cv2.imwrite("Curr_Tumor.png", curr_frame)
    #                     cv2.imwrite("Curr_Tumor_2.png", im0s)
                        
                        with open("data.csv", 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([float(depth_avg)])
#                        
#                    with open("acc.csv", 'a', newline='') as csvfile:
#                        csv_writer = csv.writer(csvfile)
#                        csv_writer.writerow([float(t_conf)])
                    
            else:
                w,h = 640, 480
                #w, h = im0s[i].shape[1], im0s[i].shape[0]
#                cv2.imwrite("Curr_Tumor.png", im0s)
                # print("width:", w, " height:", h)  # 640, 480
                curr_frame = np.array(annotator.result())
                cv2.circle(curr_frame, (w // 2, h // 2), 50, (0, 0, 255), 2)
                cv2.imwrite("Curr_Tumor.png", curr_frame)


            # Stream results
            im0 = annotator.result()




            #if self.view_img:
            #    if platform.system() == 'Linux' and p not in self.windows:
            #        self.windows.append(p)
            #        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#
#                im0 = cv2.resize(im0, (700, 400))
#                cv2.imshow(str(p), im0)
#                cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        #LOGGER.info(f"{self.s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")

        #print("--- %s seconds ---" % (time.time() - start_time))

        # Print results
        #t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        #if self.save_txt or self.save_img:
        #    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if self.save_txt else ''
        #    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        #if self.update:
        #    strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)
        


class yolov5_ros(Node):
    def __init__(self):
        super().__init__('yolov5_ros')

        self.bridge = CvBridge()

	#the image topic subscribed to is the calibrated images
        self.sub_image = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback,10)

        # parameter
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

        self.declare_parameter('weights', str(ROOT) + '/config/best_3.pt')
        self.declare_parameter('source', 0)
        self.declare_parameter('data', str(ROOT) + '/data/coco128.yaml')
        self.declare_parameter('imagez', (640,480))
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('device', '0')
        self.declare_parameter('view_img', False)
        self.declare_parameter('save_txt', False)
        self.declare_parameter('save_conf', False)
        self.declare_parameter('save_crop', False)
        self.declare_parameter('nosave', False)
        self.declare_parameter('classes', None)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('augment', False)
        self.declare_parameter('visualize', False)
        self.declare_parameter('update', False)
        self.declare_parameter('project', str(ROOT) + '/runs/detect')
        self.declare_parameter('name', 'exp')
        self.declare_parameter('exist_ok', False)
        self.declare_parameter('line_thickness', 3)
        self.declare_parameter('hide_labels', False)
        self.declare_parameter('hide_conf', False)
        self.declare_parameter('half', False)
        self.declare_parameter('dnn', False)
        self.declare_parameter('vid_stride', 1)
        self.declare_parameter('var_acc', 1)

        self.weights = self.get_parameter('weights').value
        self.source = self.get_parameter('source').value
        self.data = self.get_parameter('data').value
        self.imagez = self.get_parameter('imagez').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.view_img = self.get_parameter('view_img').value
        self.save_txt = self.get_parameter('save_txt').value
        self.save_conf = self.get_parameter('save_conf').value
        self.save_crop = self.get_parameter('save_crop').value
        self.nosave = self.get_parameter('nosave').value
        self.classes = self.get_parameter('classes').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.augment = self.get_parameter('augment').value
        self.visualize = self.get_parameter('visualize').value
        self.update = self.get_parameter('update').value
        self.project = self.get_parameter('project').value
        self.name = self.get_parameter('name').value
        self.exist_ok = self.get_parameter('exist_ok').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.hide_labels = self.get_parameter('hide_labels').value
        self.hide_conf = self.get_parameter('hide_conf').value
        self.half = self.get_parameter('half').value
        self.dnn = self.get_parameter('dnn').value
        self.vid_stride = self.get_parameter('vid_stride').value
        self.var_acc = self.get_parameter('var_acc').value

        self.yolov5 = yolov5_demo(self.weights,
                                    self.source,
                                    self.data,
                                    self.imagez,
                                    self.conf_thres,
                                    self.iou_thres,
                                    self.max_det,
                                    self.device,
                                    self.view_img,
                                    self.save_txt,
                                    self.save_conf,
                                    self.save_crop,
                                    self.nosave,
                                    self.classes,
                                    self.agnostic_nms,
                                    self.augment,
                                    self.visualize,
                                    self.update,
                                    self.project,
                                    self.name,
                                    self.exist_ok,
                                    self.line_thickness,
                                    self.hide_labels,
                                    self.hide_conf,
                                    self.half,
                                    self.dnn,
                                    self.vid_stride,
                                    self.var_acc
                                    )

    


    def image_callback(self, image:Image):
        image_raw = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.yolov5.image_callback(image_raw)


def ros_main(args=None):

    rclpy.init(args=args)
    yolov5_node = yolov5_ros()
    rclpy.spin(yolov5_node)
    
    yolov5_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    ros_main()


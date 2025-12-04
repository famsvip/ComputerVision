# This Python file uses the following encoding: utf-8
# YOLOv5 =€ by Ultralytics, AGPL-3.0 license
"""
Run YOLOv9 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
import matplotlib.pyplot as plt
import blur_det_main_ as blur

print("Done importing")

@smart_inference_mode()
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

def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        var_acc=1 #type of version
):
    
    if os.path.exists("data.csv"):
        os.remove("data.csv")

        open("data.csv", "x")
        
    if os.path.exists("acc.csv"):
        os.remove("acc.csv")

        open("acc.csv", "x")

    if os.path.exists("size_data.csv"):
        os.remove("size_data.csv")

        open("size_data.csv", "x")
    
    model_type = None
    
    if var_acc == 1:
                model_type = "MiDaS_small"
    elif var_acc == 2:
                model_type = "DPT_Hybrid"
    else:
                model_type = "DPT_Large"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = select_device(device)
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = None
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                transform = midas_transforms.dpt_transform
    else:
                transform = midas_transforms.small_transform
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    x_TOTAL_DETECT = 0
    start_time = time.time()

    for path, im, im0s, vid_cap, s in dataset:
    
#        cv2.imwrite("Curr_Tumor.png", im0s)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # print("det:", det)

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names), pil=True)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                cords = []
                conf_arr = []
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

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
#                plt.imsave("Test.png", output, cmap=cm.gray)
#                new_im = imageio.v2.imread("Test.png")
#                img_updated = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
#                img_reverted = cv2.bitwise_not(img_updated)
#                img_new = img_reverted / 255.0
                high_depth = 1.01
                t_conf = None
                curr_res = None
                for *xyxy, conf, cls in reversed(det):
                    if conf > conf_thres:
                        cords.append(xyxy)
                        conf_arr.append(conf)
                cords = [[int(value) for value in row] for row in cords]
                if len(cords):
                
                    w,h = None, None
                    if not webcam:
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                         w, h = im0s[i].shape[1], im0s[i].shape[0]
                
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
                        print("*****")
                        print(x1_t, y1_t)
                        print(x2_t, y2_t)
                        print(w,h)
                        print("*****")
                        dv = (abs(y2_t - y1_t) * abs(x2_t - x1_t)) / (w * h)
                        print("SIZE", dv)
                        dv = float(format(depth_estimate(dv), '.3f'))
                        depth_avg = dv
                        c = int(cls)
                        diameter = (abs(x1_t - x2_t) + abs(y1_t - y2_t)) // 2

                        # blur detect
                        """
                        print("im:",im)
                        image = np.array(im)
                        print("im_array:", image)
                        cv2.imwrite("saved_img_blur/saved.jpg", image)
                        if image.ndim == 4:
                            print("convert 4 to 3")
                            # convert the image from RGBA2RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                            print("converted:", image)
                        blurriness = detect_blurry(cv2.imread("saved_img_blur/saved.jpg"))
                        print("image", image)

                        blur_check = detect_blurry(cv2.imread("blur_test.jpg"))
                        print("blur check:", blur_check)
                        """
                        curr_frame = np.array(annotator.result())
                        curr_frame = curr_frame[max(y1_t-100, 0):min(y2_t+100, h), max(x1_t-100, 0):min(x2_t+100, w)]
                        cv2.imshow("cropped", curr_frame)
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
                        label = None if hide_labels else (names[c] if hide_conf else
                                                          f'Diameter:{diameter} Blur:{val:.2f} Size:{actual_size:.2f} ({low}cm-{high}cm)')
                        annotator.box_label(curr_res, label, color=(255, 0, 0))

                        print("width:", w, " height:", h)
                        radius = 50
                        curr_frame = np.array(annotator.result())
                        if x1_t < w//2 - radius < w//2 + radius < x2_t and y1_t < h//2 - radius < h//2 + radius < y2_t:
                            cv2.circle(curr_frame, (w // 2, h // 2), radius, (0, 255, 0), 2)
                        else:
                            cv2.circle(curr_frame, (w // 2, h // 2), radius, (0, 0, 255), 2)

                        #    annotator.box_label([(w // 2 - 30000 // w, h // 2 - 30000 // h), (w // 2 + 30000 // w, h // 2 + 30000 // h)],
                        #                        None, color=(0, 255, 0))

                        cv2.imwrite("Curr_Tumor.png", curr_frame)
    #                     cv2.imwrite("Curr_Tumor_2.png", im0s)
                        
                        with open("data.csv", 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([float(depth_avg)])

                        with open("size_data.csv", 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([float(actual_size)])
#                        
#                    with open("acc.csv", 'a', newline='') as csvfile:
#                        csv_writer = csv.writer(csvfile)
#                        csv_writer.writerow([float(t_conf)])
                    
            else:
                print("NA")

                if not webcam:
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    w, h = im0s[i].shape[1], im0s[i].shape[0]
#                cv2.imwrite("Curr_Tumor.png", im0s)
                # print("width:", w, " height:", h)  # 640, 480
                curr_frame = np.array(annotator.result())
                cv2.circle(curr_frame, (w // 2, h // 2), 50, (0, 0, 255), 2)
                cv2.imwrite("Curr_Tumor.png", curr_frame)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#
#                im0 = cv2.resize(im0, (700, 400))
#                cv2.imshow(str(p), im0)
#                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    print("--- %s seconds ---" % (time.time() - start_time))

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    
    print(x_TOTAL_DETECT)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--var-acc', type=int, default=1, help='acc level')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

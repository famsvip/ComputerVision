import torch

# Model
def get_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Algorithm/pt_files/best.pt')  # or yolov5m, yolov5l, yolov5x, custom
    return model

# Returns coordinates
def get_coordinates(frame, model):
    results = model(frame)                  # using the model each frame
    rows = results.pandas().xyxy[0]
    if len(rows) != 0:
        x_min, y_min, x_max,y_max = int(rows['xmin'][0]), int(rows['ymin'][0]), int(rows['xmax'][0]), int(rows['ymax'][0])
        return (x_min, y_min, x_max, y_max)
    return None

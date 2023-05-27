import os
import time

import cv2
import pandas as pd
import torch
from ultralytics import YOLO


def read_labels2dict(txt_path):
    res_dict = dict()
    with open(txt_path, 'r') as reader:
        for i, line in enumerate(reader.readlines(), start=0):
            res_dict[i] = line.strip()
    return res_dict


def create_emptydict(txt_path):
    res_dict = dict()
    with open(txt_path, 'r') as reader:
        for i, line in enumerate(reader.readlines(), start=0):
            res_dict[line.strip()] = 0
    return res_dict


# TODO think about it more
def calc_percent_garbage(counter_dict):
    n_garbage = counter_dict.get('garbage')
    n_walls = counter_dict.get('wall 0') + counter_dict.get('wall 1') + counter_dict.get('wall 2')
    if n_walls == 0:
        return 0
    return min(n_garbage*10 / n_walls * 100, 100)


# TODO think about it more
def calc_percent_kitchen(counter_dict):
    n_kitchen = counter_dict.get('kitchen')
    # counter_dict.get('wall 0') + counter_dict.get('wall 1')
    n_walls = counter_dict.get('wall 2')
    if n_walls == 0:
        return 0
    return min(n_kitchen*10 / n_walls * 100, 100)


# TODO
def calc_doors_percent(counter_dict):
    pass


def calc_elements_percent(needed_element, counter_dict):
    n_item = counter_dict.get(needed_element)
    needed_keys = counter_dict.keys()
    n_all = sum([counter_dict.get(key) for key in needed_keys if key[:4] == needed_element[:4]])
    if n_all == 0:
        return 0
    return round(n_item * 100 / n_all)


def count_items(pred_classes, items_count_dict):
    classes_to_count = ['bath', 'radiator', 'toilet', 'sink', 'shower']
    for class_name_ in classes_to_count:
        cls_id = label2id_dict.get(class_name_)
        if cls_id in pred_classes:
            if label2id_dict.get('wall 2') in pred_classes or label2id_dict.get('floor2') in pred_classes:
                items_count_dict[class_name_] += 1
            elif label2id_dict.get('wall 1') in pred_classes or label2id_dict.get('floor 1') in pred_classes:
                items_count_dict[class_name_] += 0.5
    return items_count_dict


# def calc_percent_relative(counter_dict, cls_name):
#     n_items = counter_dict.get(cls_name)
#     n_otdelok = counter_dict.get('wall 2') + counter_dict.get('ceil 1') + counter_dict.get('floor2')


# Load the YOLOv8 model
# video_name = "7.MP4"
video_name = "1.mp4"
n_seconds_timeout = 1

video_dir = "data/videos/"
video_path = os.path.join(video_dir, video_name)
labels_ttx_path = "data/labels.txt"
model = YOLO('weights/detection_best.pt')
model_cls = YOLO("weights/classify_best.pt")

# needed_rooms_list = ['zhilaya', 'koridor', 'bathroom']
labels_dict = read_labels2dict(labels_ttx_path)
label2id_dict = {v: k for k, v in labels_dict.items()}
res_dict = create_emptydict(labels_ttx_path)

items_count_dict = dict.fromkeys(['radiator', 'bath', 'toilet', 'sink', 'shower'], 0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start = time.time()
cap = cv2.VideoCapture(video_path)
num_frames_processed = 0
count_frames = 0


fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Video FPS: {fps}')
num_timeout_frames = round(n_seconds_timeout * fps)
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        if count_frames != 0:
            count_frames -= 1
            continue
        num_frames_processed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()
        print(fm)
        if (fm > 500) and (count_frames == 0):
            count_frames = num_timeout_frames
        # Run YOLOv8 inference on the frame
            cls_result = model_cls(frame)[0]
            cls_pred = cls_result.names.get(torch.argmax(cls_result.probs).item())
            if cls_pred == 'general':
                continue
            results = model.predict(frame, conf=0.4, device=device)
            for result in results:
                pred_classes = result.boxes.cls
                items_count_dict = count_items(pred_classes, items_count_dict)

                for cls_id in pred_classes:
                    class_name = labels_dict.get(cls_id.item())
                    res_dict[class_name] += 1

            # Visualize the results on the frame
            # annotated_frame = results[0].plot()
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
    else:
        break

end = time.time()
elapsed = end - start
fps = num_frames_processed / elapsed
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))
res_percents = {}
elements = ['ceil 0', 'ceil 1', 'floor 0', 'floor 1', 'floor2',
            'podokonnik 0', 'podokonnik 1', 'socket 0', 'socket',
            'wall 0', 'wall 1', 'wall 2']
for item in elements:
    res_percent = calc_elements_percent(item, res_dict)
    res_percents[item] = res_percent

res_percents['kitchen'] = calc_percent_kitchen(res_dict)
res_percents['garbage'] = calc_percent_garbage(res_dict)

elements_approx = ['radiator', 'bath', 'toilet', 'sink', 'shower']
for item in elements_approx:
    if not res_dict.get(item):
        res_percents[item] = 0
    else:
        res_percents[item] = round(items_count_dict.get(item, 0) * 100 / res_dict.get(item))
print(res_percents)
cap.release()
cv2.destroyAllWindows()

import json

out_name = os.path.splitext(video_name)[0]
os.makedirs(f'{video_dir}/timeout_{n_seconds_timeout}', exist_ok=True)
with open(f'{video_dir}/timeout_{n_seconds_timeout}/{out_name}.json', 'w') as fp:
    json.dump(res_percents, fp)


import json
import time

import cv2


file_path = "keyPoints_coordinates_for_learning.txt"
key_points_list = []

with open(file_path, "r") as file:
    for line in file:
        # 解析每个JSON对象
        key_points_data = json.loads(line)
        key_points_list.append(key_points_data)

print(len(key_points_list))
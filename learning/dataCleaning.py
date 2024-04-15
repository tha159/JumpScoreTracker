import json
import time

import cv2


def draw_skeleton(image, frame_data):
    # print(frame_data)
    # 定义连接关系
    connections = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (0, 6), (1, 7), (6, 8), (8, 10), (10, 12), (12, 14), (1, 7),
                   (7, 9), (9, 11), (11, 13), (13, 15)]

    # 绘制关键点


    for i in range(len(frame_data)):
        key_point = frame_data[i]
        x, y = key_point[1], key_point[2]
        cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)  # 画关键点

    # 绘制连接线
    # 绘制骨骼
    for connection in connections:
        start_point = frame_data[connection[0]][1:]  # 获取起始点坐标
        end_point = frame_data[connection[1]][1:]  # 获取结束点坐标
        cv2.line(image, tuple(start_point), tuple(end_point), (0, 0, 255), 2)

    return image
def saveList(newdata):
    with open("data_train2.txt", "w") as file:
        # 将列表的每个元素转换为字符串并写入文件
        for item in newdata:
            file.write(str(item) + "\n")
        print('save success!')

# 读取文件
file_path = "data_train.txt"
key_points_list = []

with open(file_path, "r") as file:
    for line in file:
        # 解析每个JSON对象
        key_points_data = json.loads(line)
        # key_points_list.append([key_points_data,0])
        key_points_list.append(key_points_data)


print(len(key_points_list))

# # 将所有帧的标记设置为0
# for frame_data in key_points_list:
#     frame_data.append([0])

frame_index = 0
while True:
    # 获取当前帧数据
    frame_data = key_points_list[frame_index][0]

    # 构造图像
    # 这里你需要自己写代码来绘制关键点
    image = cv2.imread(r'D:\user\work\sport\sport_demo1\img\2findRuler.jpg')

    # 根据 frame_data 绘制图像
    image = draw_skeleton(image, frame_data)

    # 显示图像
    cv2.imshow('Frame', image)
    # 监听键盘事件
    key = cv2.waitKey(0) & 0xFF

    # 处理键盘事件
    if key == ord('s'):  # 保存但不退出
        # 保存数据
        # 这里你需要编写代码将数据保存到文件中
        saveList(key_points_list)
    elif key == ord('0'):  # 标注为动作0:站立

        key_points_list[frame_index][1] = 0
        print(key_points_list[frame_index][1])
    elif key == ord('1'):  # 标注为动作1:下蹲
        key_points_list[frame_index][1] = 1
        print(key_points_list[frame_index][1])
    elif key == ord('2'):  # 标注为动作2：跌倒
        key_points_list[frame_index][1] = 2
        print(key_points_list[frame_index][1])
    elif key == ord('3'):  # 标注为动作3：向上跳
        key_points_list[frame_index][1] = 3
        print(key_points_list[frame_index][1])

    elif key == ord('a'):  # 方向键 ←
        frame_index -= 1
        if frame_index < 0:
            frame_index = 0
        print('previous frame',key_points_list[frame_index][1])
    elif key == ord('d'):  # 方向键 →
        frame_index += 1
        if frame_index >= len(key_points_list):
            frame_index = len(key_points_list) - 1
        print('next frame',key_points_list[frame_index][1])

    # 检查是否按下了退出键
    if key == ord('q'):
        break

cv2.destroyAllWindows()

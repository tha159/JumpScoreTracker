import sqlite3
import time
from datetime import datetime

import torch
import copy
import cv2
import math
import numpy as np
import mediapipe as mp
from PyQt5.QtGui import QImage, QPixmap

from PoseModel import KeyPointsLSTM

class Jumping():
    mpPose = None
    pose = None
    mpDraw = None
    model = []
    jumper = []
    length = 0
    poseLm_coordinates = []
    valid_frame_images = []
    foot_length = 32
    score = []
    remaining_chances = 3
    ruler_seeker = None
    draw_long_ruler = True
    keyPoints = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    keyPoints_coordinates_for_learning = []
    keyPoints_coordinates_previous = []
    keyPoints_coordinates_now = [[11, 12, 239], [12, 62, 250], [13, 8, 326], [14, 72, 334], [15, 35, 388],
                                 [16, 86, 412], [23, 13, 398], [24, 52, 399], [25, 17, 501], [26, 91, 481],
                                 [27, 35, 593], [28, 79, 580], [29, 25, 616], [30, 65, 598], [31, 70, 606],
                                 [32, 123, 596]]
    Has_entered_takeoff_zone = False
    Has_jumped = False
    gotScore = False
    fall = False
    has_in_ruler_right_area = False
    has_out_ruler_right_area = False
    is_away_ruler_area = False
    JumpOver = False
    label = ['stand', 'squat', 'fall', 'jump up']
    land_points = []
    imgHeight = 720
    imgWidth = 1280
    poseLms = []
    frameImage = []
    predicted_label_previous = 0
    predicted_label_now = 0
    highestYHeight = 0
    highestYFrame = []
    pTime = 0
    name = "Lee"

    def __init__(self, ruler_seeker,name):
        #跳远人姓名
        self.name = name
        # 加载模型
        self.load_model()
        # 加载尺子,假设开始跳远检测前先扫描尺子位置，后续检测跳远就不用执行扫描尺子操作了。
        self.ruler_seeker = ruler_seeker
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.mpDraw = mp.solutions.drawing_utils
        print('准备完成！')

    def startJumpxxx(self, remaining_chance):
        self.remaining_chances = remaining_chance
        # 打开摄像头
        # cap = cv2.VideoCapture(0)
        # 打开视频文件
        video_path = 'img/jump_for_test_2.avi'  # 视频文件的路径
        # 打开相机
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and self.remaining_chances > 0:

            # 读取帧图片
            ret, self.frameImage = cap.read()

            # 如果该帧有图片
            if ret:
                # 转变图片从bgr到rgb、
                self.frameImage = cv2.cvtColor(self.frameImage, cv2.COLOR_BGR2RGB)

                # 检测人Pose
                result = self.pose.process(self.frameImage)
                self.poseLms = result.pose_landmarks
                # 如果有人pose
                if self.poseLms:
                    # 绘制连接线，不显示端点
                    self.draw_skeleton()

                    # 准备记录关键点坐标
                    self.keyPoints_coordinates_previous = self.keyPoints_coordinates_now
                    self.keyPoints_coordinates_now = []

                    # 记录该帧关键点坐标
                    self.getKeyPointsPoseLms()

                    # 检测姿势
                    self.predicted_label_previous = self.predicted_label_now
                    self.predicted_label_now = self.pose_detect(self.keyPoints_coordinates_now)

                    # 跳远第一步：检查是否进入起跳区域
                    if self.Has_entered_takeoff_zone == False:
                        self.Has_entered_takeoff_zone = self.check_Has_entered_takeoff_zone()
                    elif self.Has_entered_takeoff_zone == True:
                        # 如果没有起跳
                        if self.Has_jumped == False:
                            # 显示已进入起跳区域
                            self.draw_jump_ready_remind()
                            # 检查是否踩线并提醒
                            self.check_if_over_line(self.keyPoints_coordinates_now)

                            # 跳远第二步：进入起跳区域则检查是否起跳
                            # 起跳检测：上一帧为下蹲，这一帧为向上跳
                            if self.predicted_label_previous == 1 and self.predicted_label_now == 3:
                                # 并检测起跳时是否踩线
                                if self.check_if_over_line(self.keyPoints_coordinates_previous):
                                    # 如果踩线，则记为本次机会成绩0分
                                    self.score.append([0, 'over line'])
                                    self.remaining_chances -= 1
                                    # 复原参数
                                    self.refresh_parameters()
                                else:
                                    self.Has_jumped = True
                        elif self.Has_jumped == True:
                            # 跳远第三步：检测是否跳入落地区域（右侧刻度线内）
                            # 如果已经进入右侧尺子区域
                            if self.has_in_ruler_right_area == False:
                                self.has_in_ruler_right_area = self.check_has_in_ruler_right_area()
                            elif self.has_in_ruler_right_area == True:
                                # 如果跳远还未结束
                                if self.JumpOver == False:
                                    # 检测是否有跌倒
                                    if self.predicted_label_now == 2:
                                        # 如果跌倒，则记为本次机会成绩0分
                                        self.score.append([0, 'fall'])
                                        self.remaining_chances -= 1
                                        # 复原参数
                                        self.refresh_parameters()
                                    else:
                                        # 找跳远落地点，最矮时为落地点
                                        height = self.getHeight(self.keyPoints_coordinates_now)
                                        if height > self.highestYHeight:
                                            self.highestYHeight = height
                                            self.highestYFrame = self.keyPoints_coordinates_now
                                        # 如果离开右侧尺子区域，则视为跳远结束
                                        self.has_out_ruler_right_area = self.check_has_out_ruler_right_area()
                                        # if self.has_out_ruler_right_area == True:
                                        if self.predicted_label_now == 0 and self.predicted_label_previous == 0:
                                            self.JumpOver = True
                                elif self.JumpOver == True:
                                    self.getJumpScore()
                                    self.remaining_chances -= 1
                                    # 复原参数
                                    self.refresh_parameters()
                    # 显示姿势标签在人的头顶
                    self.draw_label()
                else:
                    # 如果没有检测到关键点，这里不进行绘制操作
                    pass

                # 调试信息
                # self.draw_debug_info()

                # 画尺子
                self.draw_ruler(is_ruler_long=False)
                # 绘制帧率fps
                self.draw_fps()
                # 绘制着地点，方便调试
                self.draw_land_marks()
                # 绘制成绩
                self.draw_score()
                # 最终显示图片
                self.draw_image()

            if cv2.waitKey(1) == ord('0'):
                break
        if self.remaining_chances == 0:
            self.save_score()
            time.sleep(999)
    def draw_qt_image(self):
        # self.frameImage = cv2.cvtColor(self.frameImage, cv2.COLOR_RGB2BGR)
        image = self.frameImage.copy()
        h, w, c = image.shape
        qImg = QImage(image.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        return pixmap

    def draw_image(self):
        self.frameImage = cv2.cvtColor(self.frameImage, cv2.COLOR_RGB2BGR)

        cv2.imshow('img', self.frameImage)

    def save_score(self):
        scores = self.score
        # 连接到数据库（如果不存在，则会自动创建）
        conn = sqlite3.connect('jump_results.db')
        c = conn.cursor()

        # 创建数据表
        c.execute('''CREATE TABLE IF NOT EXISTS jump_results
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      score REAL,
                      status TEXT,
                      updated_at TEXT,
                      created_at TEXT)''')

        # 姓名
        name = self.name

        # 获取当前时间
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 插入数据
        for score in scores:
            c.execute(
                "INSERT INTO jump_results (name, score, status, updated_at, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, score[0], score[1], now, now))

        # 保存更改
        conn.commit()

        # 关闭连接
        conn.close()

    def draw_debug_info(self):
        info = f"Has_entered_takeoff_zone: {self.Has_entered_takeoff_zone}"
        info += f"\nHas_jumped: {self.Has_jumped}"
        info += f"\nhas_in_ruler_right_area: {self.has_in_ruler_right_area}"
        info += f"\nfall: {self.fall}"
        info += f"\nis_away_ruler_area: {self.is_away_ruler_area}"
        info += f"\nJumpOver: {self.JumpOver}"
        info += f"\ngotScore: {self.gotScore}"

        # 在图像上绘制信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 3
        text_color = (255, 255, 255)  # 白色
        x, y = 700, 70  # 文字起始坐标
        line_height = 30  # 行高

        lines = info.split('\n')
        for line in lines:
            cv2.putText(self.frameImage, line, (x, y), font, font_scale, text_color, font_thickness)
            y += line_height




    def draw_score(self):
        score_txt = 'Chance left: ' + str(self.remaining_chances)
        if self.remaining_chances <= 2:
            score_txt = score_txt + ' [1: ' + str(self.score[0][0]) + ', ' + self.score[0][1] + ']'
        if self.remaining_chances <= 1:
            score_txt = score_txt + ' [2: ' + str(self.score[1][0]) + ', ' + self.score[1][1] + ']'
        if self.remaining_chances <= 0:
            score_txt = score_txt + ' [3: ' + str(self.score[2][0]) + ', ' + self.score[2][1] + ']'
        # 显示成绩
        cv2.putText(self.frameImage, score_txt, (50, 85),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (255, 0, 0),
                    2)

    def draw_land_marks(self):
        # 打下落地标记
        if self.land_points != None:
            for mark in self.land_points:
                cv2.circle(self.frameImage, (mark[0], mark[1]), 3, (255, 255, 255), cv2.FILLED)

    def draw_fps(self):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(self.frameImage, f"FPS : {int(fps)}", (self.imgWidth - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 245, 0), 2)

    def getJumpScore(self):
        # 算出最矮时的左脚和右脚后根
        left_foot_rear_x = self.highestYFrame[-4][1]
        left_foot_rear_y = self.highestYFrame[-4][2]
        right_foot_rear_x = self.highestYFrame[-3][1]
        right_foot_rear_y = self.highestYFrame[-3][2]
        self.land_points.append([left_foot_rear_x, left_foot_rear_y])
        self.land_points.append([right_foot_rear_x, right_foot_rear_y])
        # 最后面的脚后跟为落地成绩
        left_score = self.getScore(left_foot_rear_x, left_foot_rear_y)
        right_score = self.getScore(right_foot_rear_x, right_foot_rear_y)
        score = min(left_score, right_score)
        self.score.append([score, 'success'])
        return score

    def check_is_away_ruler(self):
        pointx = self.keyPoints_coordinates_now[-1][1]
        pointy = self.keyPoints_coordinates_now[-1][2]
        area = self.ruler_seeker.ruler_right_area
        check_is_away_ruler = not (self.point_in_polygon([pointx, pointy], area))
        self.is_away_ruler_area = check_is_away_ruler
        if check_is_away_ruler:
            return True
        else:
            return False
    def check_has_out_ruler_right_area(self):
        pointx = self.keyPoints_coordinates_now[-1][1]
        pointy = self.keyPoints_coordinates_now[-1][2]
        area = self.ruler_seeker.ruler_right_area
        check_has_out_ruler_right_area = not self.point_in_polygon([pointx, pointy], area)
        if check_has_out_ruler_right_area == True:
            return True
        else:
            return False

    def check_has_in_ruler_right_area(self):
        pointx = self.keyPoints_coordinates_now[-1][1]
        pointy = self.keyPoints_coordinates_now[-1][2]
        area = self.ruler_seeker.ruler_right_area
        check_has_in_ruler_right_area = self.point_in_polygon([pointx, pointy], area)
        if check_has_in_ruler_right_area:
            return True
        else:
            return False

    def refresh_parameters(self):
        self.Has_entered_takeoff_zone = False
        self.Has_jumped = False
        self.has_in_ruler_right_area = False
        self.fall = False
        self.has_out_ruler_right_area = False
        self.JumpOver = False
        self.gotScore = False

    def check_if_over_line(self, keyPointsCoordinates):
        foot_length = self.getFootLength()
        # 只判断右脚脚尖是否在起跳区域内
        foot_points = keyPointsCoordinates[-1]
        xPos = foot_points[1]
        yPos = foot_points[2]
        if self.point_in_polygon([xPos, yPos], self.ruler_seeker.startJumpArea) == False:
            # 如果在，就文字提醒
            cv2.putText(self.frameImage, 'Over Line!', (xPos + 125, yPos - 105), cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (255, 0, 0),
                        2)
            return True
        else:
            return False

    def addJumper(self, time, poseLm_coordinates):
        self.jumper.append([time, poseLm_coordinates])

    def check_Has_entered_takeoff_zone(self):
        # 如果29,30,31,32（2个脚尖和2个脚后跟）进入起跳区域，则算作进入起跳区域
        fcount = 0
        for foot_points in self.keyPoints_coordinates_now[-4:]:
            xPos = foot_points[1]
            yPos = foot_points[2]
            if self.point_in_polygon([xPos, yPos], self.ruler_seeker.startJumpArea) == True:
                fcount += 1;
        if fcount == 4:
            return True
        else:
            return False

    def draw_jump_ready_remind(self):
        # 画出进入提醒
        foot_point = self.keyPoints_coordinates_now[-4]
        xPos = foot_point[1]
        yPos = foot_point[2]
        cv2.putText(self.frameImage, 'Entered Jump Area!', (xPos + 125, yPos - 120),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1,
                    (0, 0, 255),
                    2)

    def draw_label(self):
        cv2.putText(self.frameImage, self.label[self.predicted_label_now],
                    (self.keyPoints_coordinates_now[0][1] - 30, self.keyPoints_coordinates_now[0][2] - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    4)

    def draw_skeleton(self):
        self.mpDraw.draw_landmarks(
            self.frameImage,
            self.poseLms,
            self.mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=None,  # 不显示端点
            connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=0),
            # 设置连接线的样式
        )

    def draw_ruler(self, is_ruler_long):
        if is_ruler_long == False:
            self.ruler_seeker.draw_real_ruler_lines(self.frameImage)
        else:
            self.ruler_seeker.draw_long_real_ruler_lines(self.frameImage)

    def getKeyPointsPoseLms(self):
        for i, lm in enumerate(self.poseLms.landmark):
            xPos = int(lm.x * self.imgWidth)
            yPos = int(lm.y * self.imgHeight)
            self.poseLm_coordinates.append([xPos, yPos])

            if i in self.keyPoints:
                self.keyPoints_coordinates_now.append([i, xPos, yPos])
                cv2.circle(self.frameImage, (xPos, yPos), 5, (255, 255, 255), cv2.FILLED)
                # cv2.putText(self.frameImage, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                #             1)

    #逆时针输入才生效
    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            # Check if point is on the edge of polygon
            if ((p1y < y <= p2y) or (p2y < y <= p1y)) and (x >= min(p1x, p2x)):
                if p1y == p2y and x != max(p1x, p2x):
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def process_time_stamp(self, keyPoints_coordinates):
        # 创建副本以防止修改原始列表
        keyPoints_coordinates_copy = copy.deepcopy(keyPoints_coordinates)

        # 获取第一个子元素中的时间戳
        first_timestamp = keyPoints_coordinates_copy[0][1]

        # 遍历每个子元素，减去第一个时间戳
        for item in keyPoints_coordinates_copy:
            item[1] -= first_timestamp

        # 数据标注检测，0为普通，

        return keyPoints_coordinates_copy

    def pose_detect(self, keyPoints_coordinates_oneimg):
        frame_data = keyPoints_coordinates_oneimg
        # 准备输入数据并进行预测
        input_data = torch.tensor([frame_data], dtype=torch.float32)  # 将当前帧数据转换为张量
        with torch.no_grad():
            output = self.model(input_data)  # 使用模型进行预测

        _, predicted_label = torch.max(output, 1)  # 找到预测结果中概率最大的类别
        predicted_label = predicted_label.item()  # 将张量转换为标量
        return predicted_label

    def load_model(self):
        # 加载模型
        input_dim = 3
        hidden_dim = 128
        output_dim = 4
        self.model = KeyPointsLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.model.load_state_dict(torch.load("learning/keypoints_lstm_model2.pth"))
        self.model.eval()

    def getScore(self, x, y):
        long_real_ruler_lines = self.ruler_seeker.long_real_ruler_lines
        in_gap_count = 0

        for i in range(1, len(long_real_ruler_lines) - 1):
            # 如果点在两个刻度线构成的区域内

            p1 = [long_real_ruler_lines[i][0][0], long_real_ruler_lines[i][0][1]]
            p2 = [long_real_ruler_lines[i][1][0], long_real_ruler_lines[i][1][1]]
            p3 = [long_real_ruler_lines[i + 1][0][0], long_real_ruler_lines[i + 1][0][1]]
            p4 = [long_real_ruler_lines[i + 1][1][0], long_real_ruler_lines[i + 1][1][1]]
            is_land_point_in_this_gap = self.point_in_polygon([x, y], [p1, p2, p4, p3])

            if is_land_point_in_this_gap == True:
                left_distance = self.point_to_segment_distance(p1, p2, [x, y])
                right_distance = self.point_to_segment_distance(p3, p4, [x, y])
                # 左边线的跳远距离
                distance_big_ruler = 10 * (i - 1)
                # 按百分比获得落地点位置距离。假设每两条线距离为10cm
                if (right_distance != 0):
                    percent = left_distance / (right_distance+left_distance)
                else:
                    percent = 1

                distance_in_one_gap = 10 * percent
                distance = distance_big_ruler + distance_in_one_gap + 100
            else:
                in_gap_count += 1
        # 说明没在缝隙内，换算法
        if in_gap_count == 44:
            real_ruler_lines = self.ruler_seeker.real_ruler_lines
            shortest_dist = 99999
            shortest_dist_index = 0
            for i in range(len(real_ruler_lines)):
                dist = self.distance_between_points([x, y], [real_ruler_lines[i][0][0], real_ruler_lines[i][0][1]])
                if dist < shortest_dist:
                    shortest_dist = dist
                    shortest_dist_index = i
            distance = 10 * (shortest_dist_index - 1) + 100
        rounded_distance = round(distance, 1)
        return rounded_distance

    def distance_between_points(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def getFootLength(self):
        return self.foot_length

    def getHeight(self, keyPoints_coordinates_now):
        sum = 0
        for point in keyPoints_coordinates_now:
            _, x, y = point
            sum = sum + y
        return sum / len(keyPoints_coordinates_now)

    def point_to_segment_distance(self,point1, point2, point):
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point

        px = x2 - x1
        py = y2 - y1
        norm = px * px + py * py

        u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        distance = math.sqrt(dx * dx + dy * dy)

        return distance




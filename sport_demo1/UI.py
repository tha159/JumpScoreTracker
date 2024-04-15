import time
import random

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QDateTime
from PyQt5.QtGui import QColor, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QDesktopWidget, QMessageBox, QGraphicsDropShadowEffect

import MainWD
from Jumping import Jumping
from RulerSeeker import RulerSeeker


# 一些窗口通用方法
class BaseClass:

    # 移动窗口到中心
    def center(self):
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = int((screen.width() - size.width()) / 2)
        newTop = int((screen.height() - size.height()) / 2)

        self.move(newLeft, newTop)

    # 可移动窗口
    def mousePressEvent(self, event):  # 鼠标左键按下时获取鼠标坐标,按下右键取消
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()
            event.accept()
        elif event.button() == Qt.RightButton:
            self.m_flag = False

    def mouseMoveEvent(self, QMouseEvent):  # 鼠标在按下左键的情况下移动时,根据坐标移动界面
        try:
            if Qt.LeftButton and self.m_flag:
                self.move(QMouseEvent.globalPos() - self.m_Position)
                QMouseEvent.accept()
        except:
            pass

    def mouseReleaseEvent(self, QMouseEvent):  # 鼠标按键释放时,取消移动
        self.m_flag = False

    # 开发中提示
    def future(self):
        QMessageBox.information(self, '敬请期待', '该功能正在开发中，敬请期待！')


class loadMain(QMainWindow, MainWD.Ui_MainWindow, BaseClass):
    score = []

    def __init__(self, parent=None):
        super().__init__(parent=None)
        self.setupUi(self)  # 加载ui
        self.center()

        print("窗口加载完毕！")

        # #无框和阴影
        self.setWindowFlag(Qt.FramelessWindowHint)  # 将界面设置为无框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 将界面属性设置为透明
        self.shadow = QGraphicsDropShadowEffect()  # 设定一个阴影,半径为10,颜色为#444444,定位为0,0
        self.shadow.setBlurRadius(10)
        self.shadow.setColor(QColor("#444444"))
        self.shadow.setOffset(0, 0)

        self.pushButton.clicked.connect(self.start_test)  # 开始测试
        self.pushButton_2.clicked.connect(self.start_check)  # 开始检录

    def start_test(self):
        test_images = 'img/2findRuler.jpg'
        ruler_template = 'img/ruler_template.jpg'
        video_name = 'img/jump_for_test_2.avi'


        remaining_chance = 3
        ruler_seeker = RulerSeeker(test_images, ruler_template)
        jumping = Jumping(ruler_seeker, '李华')
        self.start_jump(jumping, remaining_chance)


    def start_check(self):
        names = [
            '张小明', '李小红', '王大伟', '刘小强', '陈小芳', '赵大涛', '周小敏', '吴小华', '徐小刚', '孙小艾',
            '朱小梅', '曹小鹏', '林小琳', '郑小雨', '唐小勇', '任小婷', '梁小敏', '蔡小静', '叶小强', '胡小丽',
            '余小杰', '夏小琴', '柯小明', '温小敏', '余小娟', '尹小静', '昌小伟', '苏小洋', '易小勇', '潘小丽',
            '龚小勇', '魏小娟', '陆小军', '钮小玲', '邹大伟', '安小明', '凌小涛', '费小静', '蒋小明',
            '张三', '李四', '王五', '李华', '刘强', '赵敏', '陈静', '周涛', '黄飞', '吴明',
            '徐娟', '孙杰', '朱琳', '曹鹏', '林伟', '郑敏', '唐勇', '任娟', '梁敏', '蔡静',
            '叶伟', '胡秀英', '余伟', '夏丽', '柯明', '温敏', '余娟', '尹静', '昌伟', '苏洋',
            '易勇', '潘丽', '龚勇', '魏娟', '陆军', '钮玲', '邹伟', '安丽', '凌涛', '费静',
            '蒋明', '柏娟', '窦勇', '施婷', '阎强', '金霞', '董军', '鲁静', '粟洋', '路静'
        ]

        random.shuffle(names)
        text = ''
        for name in names:
                text = text + name + '、'
        self.textEdit.setText(text)

    def start_jump(self, jumping, remaining_chance):
        jumping.remaining_chances = remaining_chance
        # 打开摄像头
        # cap = cv2.VideoCapture(0)
        # 打开视频文件
        video_path = 'img/jump_for_test_2.avi'  # 视频文件的路径
        # 打开相机
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and jumping.remaining_chances > 0:

            # 读取帧图片
            ret, jumping.frameImage = cap.read()

            # 如果该帧有图片
            if ret:
                # 转变图片从bgr到rgb、
                jumping.frameImage = cv2.cvtColor(jumping.frameImage, cv2.COLOR_BGR2RGB)

                # 检测人Pose
                result = jumping.pose.process(jumping.frameImage)
                jumping.poseLms = result.pose_landmarks
                # 如果有人pose
                if jumping.poseLms:
                    # 绘制连接线，不显示端点
                    jumping.draw_skeleton()

                    # 准备记录关键点坐标
                    jumping.keyPoints_coordinates_previous = jumping.keyPoints_coordinates_now
                    jumping.keyPoints_coordinates_now = []

                    # 记录该帧关键点坐标
                    jumping.getKeyPointsPoseLms()

                    # 检测姿势
                    jumping.predicted_label_previous = jumping.predicted_label_now
                    jumping.predicted_label_now = jumping.pose_detect(jumping.keyPoints_coordinates_now)

                    # 跳远第一步：检查是否进入起跳区域
                    if jumping.Has_entered_takeoff_zone == False:
                        jumping.Has_entered_takeoff_zone = jumping.check_Has_entered_takeoff_zone()
                    elif jumping.Has_entered_takeoff_zone == True:
                        # 如果没有起跳
                        if jumping.Has_jumped == False:
                            # 显示已进入起跳区域
                            jumping.draw_jump_ready_remind()
                            # 检查是否踩线    并提醒
                            jumping.check_if_over_line(jumping.keyPoints_coordinates_now)

                            # 跳远第二步：进入起跳区域则检查是否起跳
                            # 起跳检测：上一帧为下蹲，这一帧为向上跳
                            if jumping.predicted_label_previous == 1 and jumping.predicted_label_now == 3:
                                # 并检测起跳时是否踩线
                                if jumping.check_if_over_line(jumping.keyPoints_coordinates_previous):
                                    # 如果踩线，则记为本次机会成绩0分
                                    jumping.score.append([0, 'over line'])
                                    jumping.remaining_chances -= 1
                                    # 复原参数
                                    jumping.refresh_parameters()
                                    self.score.append(0)
                                else:
                                    jumping.Has_jumped = True
                        elif jumping.Has_jumped == True:
                            # 跳远第三步：检测是否跳入落地区域（右侧刻度线内）
                            # 如果已经进入右侧尺子区域
                            if jumping.has_in_ruler_right_area == False:
                                jumping.has_in_ruler_right_area = jumping.check_has_in_ruler_right_area()
                            elif jumping.has_in_ruler_right_area == True:
                                # 如果跳远还未结束
                                if jumping.JumpOver == False:
                                    # 检测是否有跌倒
                                    if jumping.predicted_label_now == 2:
                                        # 如果跌倒，则记为本次机会成绩0分
                                        jumping.score.append([0, 'fall'])
                                        jumping.remaining_chances -= 1
                                        # 复原参数
                                        jumping.refresh_parameters()
                                        self.score.append(0)
                                    else:
                                        # 找跳远落地点，最矮时为落地点
                                        height = jumping.getHeight(jumping.keyPoints_coordinates_now)
                                        if height > jumping.highestYHeight:
                                            jumping.highestYHeight = height
                                            jumping.highestYFrame = jumping.keyPoints_coordinates_now
                                        # 如果离开右侧尺子区域，则视为跳远结束
                                        jumping.has_out_ruler_right_area = jumping.check_has_out_ruler_right_area()
                                        # if self.has_out_ruler_right_area == True:
                                        if jumping.predicted_label_now == 0 and jumping.predicted_label_previous == 0:
                                            jumping.JumpOver = True
                                elif jumping.JumpOver == True:
                                    score = jumping.getJumpScore()
                                    self.score.append(score)
                                    jumping.remaining_chances -= 1
                                    # 复原参数
                                    jumping.refresh_parameters()
                    # 显示姿势标签在人的头顶
                    jumping.draw_label()
                else:
                    # 如果没有检测到关键点，这里不进行绘制操作
                    pass
                # 调试信息
                 # self.draw_debug_info()

                # 画尺子
                jumping.draw_ruler(is_ruler_long=False)
                # 绘制帧率fps
                jumping.draw_fps()
                # 绘制着地点，方便调试
                jumping.draw_land_marks()
                # 绘制成绩
                jumping.draw_score()
                # 最终显示图片
                self.label.setPixmap(jumping.draw_qt_image())
                self.label.repaint()
                score_len = len(self.score)
                if score_len == 1 :
                    self.label_11.setText(str(self.score[0]))
                elif score_len == 2:
                    self.label_11.setText(str(self.score[0]))
                    self.label_12.setText(str(self.score[1]))
                elif score_len == 3:
                    self.label_11.setText(str(self.score[0]))
                    self.label_12.setText(str(self.score[1]))
                    self.label_13.setText(str(self.score[2]))


        if jumping.remaining_chances != 0:
            # 最终显示图片
            self.label.setPixmap(jumping.draw_qt_image())
            self.label.repaint()

            jumping.save_score()

            # 释放视频对象
            cap.release()

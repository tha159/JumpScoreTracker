import cv2
import mediapipe as mp
import time

#打开摄像头
# cap = cv2.VideoCapture(0)
# 打开视频文件
video_path = '../img/jump_for_test_2.avi'  # 视频文件的路径
cap = cv2.VideoCapture(video_path)

mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.99)
print(pose)
mpDraw = mp.solutions.drawing_utils
poseLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
poseConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)
pTime = 0
cTime = 0
keyPoints = (11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32)

while True:
    ret, img = cap.read()

    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        print(result.pose_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        poseLms = result.pose_landmarks

        if poseLms:
            # 绘制连接线，不显示端点
            mpDraw.draw_landmarks(
                img,
                poseLms,
                mpPose.POSE_CONNECTIONS,
                landmark_drawing_spec=None,  # 不显示端点
                connection_drawing_spec=mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=0),  # 设置连接线的样式
            )
            for i, lm in enumerate(poseLms.landmark):
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)
                if i in keyPoints:
                    cv2.circle(img, (xPos, yPos), 5, (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, str(i), (xPos - 25, yPos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            # 如果没有检测到关键点，这里不进行绘制操作
            pass
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}",(imgWidth-100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 245, 0), 2)

        cv2.imshow('img', img)


    if cv2.waitKey(1) == ord('0'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
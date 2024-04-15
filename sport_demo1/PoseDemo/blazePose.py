import cv2
import mediapipe as mp

# 初始化MediaPipe Pose模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 读取视频
cap = cv2.VideoCapture('your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换BGR图像为RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 姿势估计
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # 可视化关键点
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # 显示结果
    cv2.imshow('MediaPipe BlazePose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

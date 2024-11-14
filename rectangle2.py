import cv2
import numpy as np

def find_and_complete_rectangle(frame):
    thresh = 500  # 可调整的边缘检测阈值
    N = 5  # 尝试的阈值级别数量

    out = frame.copy()

    # 转换为灰度图像并应用中值滤波
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)

    for l in range(N):
        if l == 0:
            gray_edge = cv2.Canny(gray, 5, thresh)
            gray_edge = cv2.dilate(gray_edge, None)
        else:
            _, gray_edge = cv2.threshold(gray, (l + 1) * 255 // N, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(gray_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # 近似多边形
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

            # 检查是否为三角形或矩形
            if len(approx) == 3 or len(approx) == 4:
                if cv2.isContourConvex(approx):
                    # 如果是三角形，估计第四个点
                    if len(approx) == 3:
                        approx = estimate_missing_point(approx)

                    # 绘制矩形
                    cv2.polylines(out, [approx], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    return out

def estimate_missing_point(approx):
    # 根据三个角点估计第四个角点
    pts = [point[0] for point in approx]
    a, b, c = pts[0], pts[1], pts[2]

    # 使用向量法估计缺失的第四个点 d
    d = c + (b - a)
    return np.array([a, b, c, d], dtype=np.int32).reshape(-1, 1, 2)

# 捕捉视频
cap = cv2.VideoCapture(0)  # 0为默认摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频流！")
        break

    # 处理每帧图像
    output_frame = find_and_complete_rectangle(frame)

    # 显示处理后的帧
    cv2.imshow("Completed Rectangle", output_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
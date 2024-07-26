import cv2  
import numpy as np  
  
# 打开默认摄像头  
cap = cv2.VideoCapture(0)  
  
if not cap.isOpened():  
    print("Error: Unable to open camera.")  
    exit()  

def nothing(x):
    pass

# 显示结果  
# 创建一个名为 'original' 的窗口
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
# 调整 'original' 窗口的大小
cv2.resizeWindow('original', 600, 600)

# 创建一个名为 'binary' 的窗口
cv2.namedWindow('binary', cv2.WINDOW_NORMAL)
# 调整 'binary' 窗口的大小
cv2.resizeWindow('binary', 600, 600)

cv2.createTrackbar('Threshold1', 'binary', 0, 255, nothing)
cv2.createTrackbar('Threshold2', 'binary', 0, 255, nothing)
  
while True:  
    # 读取一帧  
    ret, frame = cap.read()  
    if not ret:  
        print("Error: Unable to read camera frame.")  
        break  

    threshold1 = cv2.getTrackbarPos('Threshold1', 'binary')
    threshold2 = cv2.getTrackbarPos('Threshold2', 'binary')

    _, bin_frame = cv2.threshold(frame, threshold1, threshold2, cv2.THRESH_BINARY_INV)
    
    # # 转换为灰度图  
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
  
    # # 应用高斯模糊以去除噪声  
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
  
    # # 边缘检测  
    # edged = cv2.Canny(blurred, 30, 150)  
  
    # # 查找轮廓  
    # contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
  
    # # 遍历轮廓  
    # for cnt in contours:  
    #     # 轮廓近似  
    #     epsilon = 0.04 * cv2.arcLength(cnt, True)  
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)  
  
    #     # 检查是否为四边形（矩形或正方形）  
    #     if len(approx) == 4:  
    #         # 计算边界框  
    #         x, y, w, h = cv2.boundingRect(approx)  
  
    #         # 绘制轮廓和边界框  
    #         cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)  
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
  

    cv2.imshow('original', frame)  
    cv2.imshow('binary', bin_frame)  
  
    # 按 'q' 键退出  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# 释放摄像头资源并关闭所有窗口  
cap.release()  
cv2.destroyAllWindows()
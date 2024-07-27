import cv2
import numpy as np
import math
import serial
import struct



cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

CAP_WIDTH = 1920
CAP_HEIGHT = 1080
# ROI
CAP_CENTER = (1080, 607)
CAP_LENGTH = 800

BOX_SIZE_RATIO = 0.1
EPSILON_RATIO = 0.05

color = {'red': (0, 0, 255),
         'green': (0, 255, 0),
         'blue': (255, 0, 0),
         'yellow': (0, 255, 255), }

# 4 members: red in white, red in black, green in white, green in black
color_range = {'color_red_in_white': {'Lower': np.array([0, 40, 240]), 'Upper': np.array([40, 255, 255])},
               'color_red_in_black': {'Lower': np.array([0, 40, 100]), 'Upper': np.array([40, 255, 250])},
               'color_green_in_white': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 254])},
               'color_green_in_black': {'Lower': np.array([60, 40, 100]), 'Upper': np.array([100, 255, 250])}}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)  # 宽度
CAP_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)  # 宽度
CAP_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 宽度
# cap.set(cv2.CAP_PROP_FPS, 30)	            # 帧率 帧/秒
cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)  # 亮度
cap.set(cv2.CAP_PROP_CONTRAST, 50)  # 对比度
cap.set(cv2.CAP_PROP_SATURATION, 100)  # 饱和度
cap.set(cv2.CAP_PROP_HUE, 20)  # 色调 50
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # 曝光
print("SIZE:")
print((CAP_WIDTH, CAP_HEIGHT))


class BOX:
    def __init__(self, _box, _left_up_point, _right_up_point, _left_down_point, _right_down_point):
        self.left_up_point = _left_up_point
        self.right_up_point = _right_up_point
        self.left_down_point = _left_down_point
        self.right_down_point = _right_down_point


def findBox(img_bin, frame=None):
    if img_bin is None:
        print("img_bin cannot be None")
        assert True
    min_area = int(BOX_SIZE_RATIO * len(img_bin) * len(img_bin[0]))

    ret = dict()
    cnts, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(cnts) == 0:
        return None
    for i in range(0, len(cnts)):
        if cv2.contourArea(cnts[i]) >= min_area:
            epsilon = EPSILON_RATIO * cv2.arcLength(cnts[i], True)
            approx = cv2.approxPolyDP(cnts[i], epsilon, True)

            if len(approx) == 4:
                ret[i] = approx

    for i, cnt in enumerate(cnts):
        # 如果他是个矩形而且他有爹而且爹也是个矩形
        if i in ret.keys() and len(ret[i]) == 4 and hierarchy[0][i][3] != -1 and hierarchy[0][i][
            3] in ret.keys() and len(ret[hierarchy[0][i][3]]) == 4 and hierarchy[0][i][2] == -1:
            if frame is not None:
                cv2.drawContours(frame, [np.intp(cnts[i])], -1, color['red'], 2)
                cv2.drawContours(frame, [np.intp(cnts[hierarchy[0][i][3]])], -1, color['red'], 2)
                cv2.polylines(frame, ret[i], True, color['yellow'], 10)
                cv2.polylines(frame, ret[hierarchy[0][i][3]], True, color['yellow'], 10)
                cv2.imshow("BOX", frame)

            return ret[hierarchy[0][i][3]], ret[i]


def midpoint_detection(img):
    boxs = findBox(frame_bin, img)
    if boxs is None:
        print("No contours!")
        # continue
    else:
        # 匹配角点
        midpoint = np.zeros((4, 2))
        paired = [0, 0, 0, 0]
        cnt = 0
        for i in range(0, 4):
            min_pos = None
            min_dis = 9999
            for j in range(0, 4):
                if paired[j]:
                    continue
                dis = abs(boxs[0][i][0][0] - boxs[1][j][0][0]) + abs(boxs[0][i][0][1] - boxs[1][j][0][1])
                if dis < min_dis:
                    min_dis = dis
                    min_pos = j
            if min_pos is not None:
                paired[min_pos] = 1
                midpoint[i] = (
                    boxs[0][i][0][0] + boxs[1][min_pos][0][0], boxs[0][i][0][1] + boxs[1][min_pos][0][1])
                cnt = cnt + 1
            else:
                assert True
        if cnt != 4:
            return None

        midpoint_sorted = np.zeros_like(midpoint)
        visited = [0, 0, 0, 0]
        element_next = 0
        element_now = 0
        cnt = 0
        while cnt < 4:
            cnt = cnt + 1
            element_now = element_next
            element_next = None
            min_dis = 9999
            for j in range(0, 4):
                if element_now == j or visited[j] == 1:
                    continue
                temp = math.sqrt((midpoint[element_now][0] - midpoint[j][0]) * (
                        midpoint[element_now][0] - midpoint[j][0]) + (
                                         midpoint[element_now][1] - midpoint[j][1]) * (
                                         midpoint[element_now][1] - midpoint[j][1]))
                if temp < min_dis:
                    min_dis = temp
                    element_next = j
            if element_next is not None:
                midpoint_sorted[cnt - 1] = midpoint[element_now]
            else:
                return None
        return midpoint_sorted






while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if frame is not None:
            # 获取图像
            frame = frame[int(CAP_CENTER[1] - (CAP_LENGTH / 2)):int(CAP_CENTER[1] + (CAP_LENGTH / 2)),
                    int(CAP_CENTER[0] - (CAP_LENGTH / 2)):int(CAP_CENTER[0] + (CAP_LENGTH / 2)), :]
            cv2.imshow('camera', frame)

            # 预处理
            gs_frame = cv2.GaussianBlur(frame, (7, 7), 0)  # 高斯模糊
            # cv2.imshow('gs_frame', gs_frame)
            frame_gray = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2GRAY)  # 转化成GRAY图像
            _, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            cv2.imshow('frame_bin', frame_bin)

            # 寻找矩形
            midpoint = midpoint_detection(frame)
            print(midpoint)

            # cv2.line(frame, (np.intp(midpoint_sorted[0][0]), np.intp(midpoint_sorted[0][1])),
            #          (np.intp(midpoint_sorted[1][0]), np.intp(midpoint_sorted[1][1])),
            #          (255, 255, 255))
            # cv2.line(frame, (np.intp(midpoint_sorted[1][0]), np.intp(midpoint_sorted[1][1])),
            #          (np.intp(midpoint_sorted[2][0]), np.intp(midpoint_sorted[2][1])),
            #          (255, 255, 255))
            # cv2.line(frame, (np.intp(midpoint_sorted[2][0]), np.intp(midpoint_sorted[2][1])),
            #          (np.intp(midpoint_sorted[3][0]), np.intp(midpoint_sorted[3][1])),
            #          (255, 255, 255))
            # cv2.line(frame, (np.intp(midpoint_sorted[3][0]), np.intp(midpoint_sorted[3][1])),
            #          (np.intp(midpoint_sorted[0][0]), np.intp(midpoint_sorted[0][1])),
            #          (255, 255, 255))
            # cv2.imshow('result', frame)

            # box_father = BOX(boxs[0])
            # box_son = BOX(boxs[1])
            # left_up_midpoint = (int((box_father.left_up_point[0]+box_son.left_up_point[0])/2), int((box_father.left_up_point[1]+box_son.left_up_point[1])/2))
            # right_up_midpoint = (int((box_father.right_up_point[0]+box_son.right_up_point[0])/2), int((box_father.right_up_point[1]+box_son.right_up_point[1])/2))
            # left_down_midpoint = (int((box_father.left_down_point[0]+box_son.left_down_point[0])/2), int((box_father.left_down_point[1]+box_son.left_down_point[1])/2))
            # right_down_midpoint = (int((box_father.right_down_point[0]+box_son.right_down_point[0])/2), int((box_father.right_down_point[1]+box_son.right_down_point[1])/2))
            # cv2.line(frame, left_up_midpoint, right_up_midpoint, (0, 0, 255), 1)
            # cv2.line(frame, right_up_midpoint, right_down_midpoint, (0, 0, 255), 1)
            # cv2.line(frame, right_down_midpoint, left_down_midpoint, (0, 0, 255), 1)
            # cv2.line(frame, left_down_midpoint, left_up_midpoint, (0, 0, 255), 1)
            # cv2.imshow('result', frame)

            # for i, img in enumerate(img_list):
            #     if i%2 == 1:
            #         if img_list[i-1].maxcnt is not None:
            #             continue
            #
            #     cnts, hierarchy = cv2.findContours(img.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            #     max_cnt = None
            #     max_cnt_area = 0
            #     for j in range(0, len(cnts)):
            #         if hierarchy[0][j][3] != -1:
            #             cnt_area = cv2.contourArea(cnts[j])
            #             if cnt_area > max_cnt_area:
            #                 max_cnt = cnts[j]
            #                 max_cnt_area = cnt_area
            #     if max_cnt is None:
            #         print("No"+ img.name)
            #     else:
            #         print(img.name)
            #         img.maxcnt = max_cnt
            #         img.center, img.radius = cv2.minEnclosingCircle(img.maxcnt)
            #         frame = cv2.circle(frame, (int(img.center[0]), int(img.center[1])), int(img.radius), img.color, 2)
            #         frame = cv2.circle(frame, (int(img.center[0]), int(img.center[1])), 1, img.color, 2)

            # cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            #
            # maxarea = 0
            # maxcnt = None
            # for cnt in cnts:
            #     area = cv2.contourArea(cnt)
            #     if area > maxarea:
            #         maxcnt = cnt
            #         maxarea = area
            #
            # if maxcnt is not None:
            #     maxcnt = max(cnts, key=cv2.contourArea)
            #     cv2.drawContours(frame, [np.int0(maxcnt)], -1, (0, 255, 255), 2)
            #     (x, y), r = cv2.minEnclosingCircle(maxcnt)
            #     frame = cv2.circle(frame, (int(x), int(y)), int(r), (255, 255, 0), 2)
            #     print(x, y)
            #     #str = struct.pack("1B", int())
            #     x_h = int(x / 256)
            #     x_l = int(x % 256)
            #     y_h = int(y / 256)
            #     y_l = int(y % 256)
            #     str = struct.pack("14B", 0xFF, 1, 0, x_h, x_l, y_h, y_l, 0, 0, 0, 0,0,0x0D, 0x0A)
            #     print(str)
            #     com.write(str)

            # cv2.imshow("processed frame", frame)
            # print()

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.imwrite("img.jpg", frame)
                break
            if key & 0xFF == ord('f'):
                cv2.imwrite("img.jpg", frame)
                break
        else:
            print("无画面")
    else:
        print("无法读取摄像头！")

cap.release()
cv2.destroyAllWindows()

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

// 初始化HSV阈值的上下限
int lowH = 0, lowS = 0, lowV = 0;
int highH = 180, highS = 255, highV = 255;

// 一次测试比较好的参数

#define HEIGHT 480

// 回调函数，用于Trackbar的更新
void on_low_H_thresh_trackbar(int, void *) {}
void on_high_H_thresh_trackbar(int, void *) {}
void on_low_S_thresh_trackbar(int, void *) {}
void on_high_S_thresh_trackbar(int, void *) {}
void on_low_V_thresh_trackbar(int, void *) {}
void on_high_V_thresh_trackbar(int, void *) {}

int main(int argc, char *argv[])
{
    // 检查是否提供了摄像头路径

    // argv: executable file, camera path, exposure value
    if (argc != 3)
    {
        std::cout << "用法: " << argv[0] << " <摄像头路径>" << std::endl;
        return -1;
    }

    // 打开摄像头
    cv::VideoCapture cap(argv[1],cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        std::cout << "无法打开摄像头: " << argv[1] << std::endl;
        return -1;
    }

    // cap.set(cv::CAP_PROP_EXPOSURE, 50);

    // 设置图像的宽度
    bool success1 = cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    if (!success1)
    {
        std::cout << "无法设置图像的宽度" << std::endl;
        return -1;
    }

    // // manual exposure
    // cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
    // double exposure = std::stod(argv[2]);
    // bool success4 = cap.set(cv::CAP_PROP_EXPOSURE, exposure);
    // if (!success4) {
    //   std::cout << "无法设置曝光值" << std::endl;
    //   return -1;
    // }

    // cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);		// 注意禁用的参数是0.25
    // cap.set(cv::CAP_PROP_EXPOSURE, exposure);

    // the default value the 230
    double exposure = 230;
    exposure = std::stod(argv[2]);
    // exposure = 4;
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 3.0); // 先打开自动曝光
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1.0); // 再关闭自动曝光
    cap.set(cv::CAP_PROP_EXPOSURE, exposure); // 最后设置曝光参数

    // // auto exposure
    // bool success = cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);
    // if (!success)
    // {
    //     std::cout << "无法设置自动曝光" << std::endl;
    //     return -1;
    // }

    // 设置图像的高度
    bool success2 = cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    if (!success2)
    {
        std::cout << "无法设置图像的高度" << std::endl;
        return -1;
    }

    // 设置摄像头帧率
    bool success3 = cap.set(cv::CAP_PROP_FPS, 40);
    if (!success3)
    {
        std::cout << "无法设置摄像头帧率" << std::endl;
        return -1;
    }

    // 创建窗口
    cv::namedWindow("Binary Image", cv::WINDOW_AUTOSIZE);

    // 创建Trackbars
    cv::createTrackbar("Low H", "Binary Image", &lowH, 180, on_low_H_thresh_trackbar);
    cv::createTrackbar("High H", "Binary Image", &highH, 180, on_high_H_thresh_trackbar);
    cv::createTrackbar("Low S", "Binary Image", &lowS, 255, on_low_S_thresh_trackbar);
    cv::createTrackbar("High S", "Binary Image", &highS, 255, on_high_S_thresh_trackbar);
    cv::createTrackbar("Low V", "Binary Image", &lowV, 255, on_low_V_thresh_trackbar);
    cv::createTrackbar("High V", "Binary Image", &highV, 255, on_high_V_thresh_trackbar);

    // 设置一些 文字 的参数

    // 设置文本的位置
    cv::Point text_position(10, 50); // 在图像的左上角

    // 设置文本的字体和大小
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1;
    int thickness = 2;

    // cv::SimpleBlobDetector blob_detector;
    while (true)
    {
        cv::Mat frame_raw;
        // 读取摄像头的帧
        cap >> frame_raw;
        if (frame_raw.empty())
        {
            std::cout << "无法从摄像头读取帧" << std::endl;
            break;
        }

        // select the ROI region 
        cv::Rect roi(150,150,300,180);
        cv::Mat frame = frame_raw(roi);

        // smooth the image
        cv::GaussianBlur(frame, frame, cv::Size(5, 5), 0);

        // 转换到HSV颜色空间
        cv::Mat hsv_image;
        cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

        // 定义白色的HSV范围
        cv::Scalar lower_white(lowH, lowS, lowV);
        cv::Scalar upper_white(highH, highS, highV);

        // 应用掩膜以仅保留白色区域
        cv::Mat mask;
        cv::inRange(hsv_image, lower_white, upper_white, mask);

        // 二值化处理
        cv::Mat binary_image;
        cv::bitwise_and(frame, frame, binary_image, mask);

        // 将结果图像转换为二值图像
        cv::cvtColor(binary_image, binary_image, cv::COLOR_BGR2GRAY);
        cv::imshow("grayscale", binary_image);

        // 这个函数的第三个参数thres是需要更改的，对白色的线的检测可以稍微给大一点（滤除多一点假的白点）

        
        cv::threshold(binary_image, binary_image, 120, 255, cv::THRESH_BINARY);

        // 找到轮廓
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary_image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


        // 画出轮廓
        cv::Mat drawing = cv::Mat::zeros(binary_image.size(), CV_8UC3);
        for (size_t i = std::max(0, static_cast<int>(contours.size()) - 2); i < contours.size(); i++)
        {
            cv::Scalar color = cv::Scalar(0, 255, 0);
            cv::drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
        }

        // std::cout << contours << s

        // cv
        
        cv::imshow("Binary Image", binary_image);
        cv::imshow("Original Image", frame);
        cv::imshow("draw counters", drawing);

        // 按下Esc键退出循环
        char key = (char)cv::waitKey(30);
        if (key == 27)
            break;
    }

    // 释放摄像头
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
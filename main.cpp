#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <vector>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
int img_width = 512;
int img_height = 512;

int main(int argc, char **argv)
{
    if (argc != 2)
    {

        std::cout << "Usage:" << argv[1] << "<image path>" << std::endl;
        return -1;
    }
    const char *img_path = argv[1];
    // creat face detecter
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();
    // create landmarks landmark_predictor
    dlib::shape_predictor landmark_predictor;
    // laod landmarks
    dlib::deserialize("../shape_predictor_81_face_landmarks.dat") >> landmark_predictor;
    cv::Mat orign_img = cv::imread(img_path, 1);
    cv::Mat orign_img_copy = orign_img.clone();
    // from cv::Mat to dlib bgr_pixel type
    dlib::cv_image<dlib::bgr_pixel> img(orign_img);
    cv::imshow("oring_img", orign_img);

    // do face detect
    std::vector<dlib::rectangle> faces = face_detector(img);
    for (const auto &face : faces)
    {
        // get face bbox axis
        dlib::rectangle cvRect(face.left(), face.top(), face.right(), face.bottom());
        // cv::rectangle(orign_img, cvRect, cv::Scalar(0, 255, 0), 2);
        // create crop image dlib type
        dlib::array2d<dlib::bgr_pixel> croppedDlibImage;
        // crop face image
        dlib::assign_image(croppedDlibImage, dlib::sub_image(img, cvRect));
        // from dlib typr to Mat type
        cv::Mat croppedImage = dlib::toMat(croppedDlibImage);
        cv::resize(croppedImage, croppedImage, cv::Size(orign_img.rows, orign_img.cols), cv::INTER_CUBIC);
        cv::imshow("face detect", croppedImage);
        // fet landmarks
        dlib::full_object_detection landmarks = landmark_predictor(img, face);

        // Loop through the 68 landmarks and draw them
        for (int i = 0; i < landmarks.num_parts(); i++)
        {
            dlib::point landmark = landmarks.part(i);
            cv::circle(orign_img, cv::Point(landmark.x(), landmark.y()), 2, cv::Scalar(0, 0, 255), -1);
        }
        dlib::point right_chin = landmarks.part(4);
        dlib::point right_mount = landmarks.part(48);
        dlib::point left_chin = landmarks.part(12);
        dlib::point left_mount = landmarks.part(54);
        double right_distance = std::sqrt(std::pow(right_chin.x() - right_mount.x(), 2) + std::pow(right_chin.y() - right_mount.y(), 2));
        double left_distance = std::sqrt(std::pow(left_chin.x() - left_mount.x(), 2) + std::pow(left_chin.y() - left_mount.y(), 2));
        std::cout << "left:" << left_distance << std::endl;
        std::cout << "right:" << right_distance << std::endl;
        double value = right_distance - left_distance;
        std::cout
            << "value:" << std::showpos << value << std::endl;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 1;
        cv::Scalar fontColor(10, 120, 10); // BGR color
        int fontThickness = 1;
        cv::Point textPosition(10, 30);
        cv::String main_face = "main face";
        cv::String left_face = "left face";
        cv::String right_face = "right face";
        if (value < -10.0)
        {
            cv::putText(orign_img_copy, left_face, textPosition, fontFace, fontScale, fontColor, fontThickness);
        }
        else if (value > 10.0)
        {
            cv::putText(orign_img_copy, right_face, textPosition, fontFace, fontScale, fontColor, fontThickness);
        }
        else
        {
            cv::putText(orign_img_copy, main_face, textPosition, fontFace, fontScale, fontColor, fontThickness);
        }
    }

    cv::resize(orign_img_copy, orign_img_copy, cv::Size(img_width, img_height), cv::INTER_CUBIC);
    cv::imshow("face", orign_img_copy);

    // Display the image with landmarks
    cv::imshow("Facial Landmarks", orign_img);
    cv::waitKey(0);
    cv::destroyAllWindows;
}
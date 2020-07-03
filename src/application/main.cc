#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>
#include "orb_slam/include/System.h"
#include "orb_slam/include/FPGA.h"

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
using std::placeholders::_1;

#if USE_RECONOS == 1

extern "C" {
    #include "reconos.h"
    #include "reconos_app.h"
}

#endif

#define COMPILEDWITHC11

using namespace std;


class ros2_orbslam : public rclcpp::Node
{

private:
string path_voc;
string path_settings;
cv::Mat iRGB, iDepth;

uint32_t bColorImageSet;
uint32_t bDepthImageSet;
    void color_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
      			
		const Mat tImageRGB(msg->height, msg->width, CV_8UC3, (void*)&msg->data[0]);
		cvtColor(tImageRGB, iRGB, CV_RGB2BGR);
        bColorImageSet = 1;
    }
    
    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
      iDepth = cv::Mat(msg->height, msg->width, CV_16UC1, (void*)&msg->data[0]);
      bDepthImageSet = 1;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;

public:

ros2_orbslam(string _path_voc, string _path_settings)
    : Node("ros_orbslam")
    {
      color_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "image_raw", 10, std::bind(&ros2_orbslam::color_callback, this, _1));
      depth_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "depth_raw", 10, std::bind(&ros2_orbslam::depth_callback, this, _1));
    }

int process()
{
    double tframe;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(path_voc, path_settings,ORB_SLAM2::System::RGBD,true);

     // Main loop
    
    for(int ni=0; ni<10000; ni++)
    {
        while((bDepthImageSet==0) || (bColorImageSet == 0));
        // Pass the images to the SLAM system
        SLAM.TrackRGBD(iRGB,iDepth,tframe);
        bDepthImageSet = 0;
        bColorImageSet = 0;
        tframe+= 0.010;



    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}



};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ros2_orbslam>(argv[1], argv[2]));
  rclcpp::shutdown();
  return 0;
}
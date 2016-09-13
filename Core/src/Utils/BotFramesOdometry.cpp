#include "BotFramesOdometry.h"

BotFramesOdometry::BotFramesOdometry(BotFrames* botframes_, bool useVicon, std::string cameraFrame, std::string worldFrame, std::string pelvisFrame, std::string viconFrame)
 : botFrames(botframes_),
   prevPosition(Eigen::Isometry3f::Identity()),
   useVicon(useVicon),
   cameraFrame(cameraFrame),
   worldFrame(worldFrame),
   pelvisFrame(pelvisFrame),
   viconFrame(viconFrame) {}

BotFramesOdometry::~BotFramesOdometry()
{
}

void BotFramesOdometry::initialisePose(Eigen::Matrix4f & pose, uint64_t timestamp) {
    double curr_position_camera[16];
    double curr_position_vicon[16];

    int statusCamera = 1;
    int statusVicon = 1;

    if (useVicon) {
        statusCamera = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  pelvisFrame.c_str(), timestamp, curr_position_camera);
        statusVicon = bot_frames_get_trans_mat_4x4_with_utime(botFrames, viconFrame.c_str(),  worldFrame.c_str(), timestamp, curr_position_vicon);
    } else {
        statusCamera = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  worldFrame.c_str(), timestamp, curr_position_camera);
    }

    if (!statusCamera || !statusVicon)
    {
        std::cout << "BotFramesOdometry: bot_frames returned false";
        return;
    }

    Eigen::Isometry3f currPositionCAM = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f currPositionVICON = Eigen::Isometry3f::Identity();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            currPositionCAM(i,j) = float(curr_position_camera[i*4+j]);
            currPositionVICON(i,j) = float(curr_position_vicon[i*4+j]);
        }
    }

    if (useVicon) {
        prevPosition = currPositionVICON * currPositionCAM;
    } else {
        prevPosition = currPositionCAM;
    }

    pose.topRightCorner(3,1) = prevPosition.translation();
    pose.topLeftCorner(3,3)  = prevPosition.rotation();
}

void BotFramesOdometry::getIncrementalTransformation(Eigen::Matrix4f & deltaMotion, uint64_t timestamp)
{
    double curr_position_camera[16];
    double curr_position_vicon[16];

    int statusCamera = 1;
    int statusVicon = 1;

    if (useVicon) {
        statusCamera = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  pelvisFrame.c_str(), timestamp, curr_position_camera);
        statusVicon = bot_frames_get_trans_mat_4x4_with_utime(botFrames, viconFrame.c_str(),  worldFrame.c_str(), timestamp, curr_position_vicon);
    } else {
        statusCamera = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  worldFrame.c_str(), timestamp, curr_position_camera);
    }

    if (!statusCamera || !statusVicon)
    {
        std::cout << "BotFramesOdometry: bot_frames returned false";
        return;
    }

    Eigen::Isometry3f currPositionVICON = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f currPositionCAM = Eigen::Isometry3f::Identity();


    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            currPositionCAM(i,j) = float(curr_position_camera[i*4+j]);
            currPositionVICON(i,j) = float(curr_position_vicon[i*4+j]);
        }
    }

    if (useVicon) {
        currPosition = currPositionVICON * currPositionCAM;
    } else {
        currPosition = currPositionCAM;
    }

    Eigen::Isometry3f motion = prevPosition.inverse() * currPosition;

    deltaMotion.topRightCorner(3,1) = motion.translation();
    deltaMotion.topLeftCorner(3,3)  = motion.rotation();

    prevPosition = currPosition;
}

Eigen::MatrixXd BotFramesOdometry::getCovariance()
{
    // Create an arbirary covariance:
    Eigen::MatrixXd cov(6, 6);
    cov.setIdentity();
    cov(0, 0) = 1e-06;
    cov(1, 1) = 1e-06;
    cov(2, 2) = 1e-06;
    cov(3, 3) = 1e-06;
    cov(4, 4) = 1e-06;
    cov(5, 5) = 1e-06;
    return cov;
}

void BotFramesOdometry::reset() {
    prevPosition = Eigen::Isometry3f::Identity();
}

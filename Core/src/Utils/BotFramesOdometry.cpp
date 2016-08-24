#include "BotFramesOdometry.h"

BotFramesOdometry::BotFramesOdometry(BotFrames* botframes_, std::string cameraFrame, std::string worldFrame)
 : botFrames(botframes_),
   prevPosition(Eigen::Isometry3f::Identity()),
   cameraFrame(cameraFrame),
   worldFrame(worldFrame) {}

BotFramesOdometry::~BotFramesOdometry()
{
}

void BotFramesOdometry::initialisePose(Eigen::Matrix4f & pose, uint64_t timestamp) {
    double curr_position_array[16];
    int status = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  worldFrame.c_str(), timestamp, curr_position_array);

    if (!status)
    {
        std::cout << "BotFramesOdometry: bot_frames returned false";
        return;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            prevPosition(i,j) = float(curr_position_array[i*4+j]);
            pose(i,j) = float(curr_position_array[i*4+j]);
        }
    }
}

void BotFramesOdometry::getIncrementalTransformation(Eigen::Matrix4f & deltaMotion, uint64_t timestamp)
{
    double curr_position_array[16];
    int status = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  worldFrame.c_str(), timestamp, curr_position_array);

    if (!status)
    {
        std::cout << "BotFramesOdometry: bot_frames returned false";
        return;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            currPosition(i,j) = float(curr_position_array[i*4+j]);
        }
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

#include "BotFramesOdometry.h"

BotFramesOdometry::BotFramesOdometry(BotFrames* botframes_, std::string lcmChannel)
 : botFrames(botframes_),
   prevPosition(Eigen::Isometry3f::Identity()),
   cameraFrame(lcmChannel),
   isInitialised(false) {}

BotFramesOdometry::~BotFramesOdometry()
{
}

void BotFramesOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                     Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                     uint64_t timestamp
                                                     )
{
    double curr_position_array[16];
    int status = bot_frames_get_trans_mat_4x4_with_utime(botFrames, cameraFrame.c_str(),  "local", timestamp, curr_position_array);

    if (!status)
    {
        std::cout << "BotFramesOdometry: bot_frames returned false";
        return;
    }

    Eigen::Isometry3f currPosition;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            currPosition(i,j) = float(curr_position_array[i*4+j]);
        }
    }

    if (!isInitialised) {
        prevPosition = currPosition;
        isInitialised = true;
        return;
    }

    Eigen::Isometry3f deltaMotion = prevPosition.inverse() * currPosition;

    trans = deltaMotion.translation();
    rot = deltaMotion.rotation();

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
    isInitialised  = false;
}

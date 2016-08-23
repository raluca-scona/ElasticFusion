#ifndef BOTFRAMESODOMETRY_H_
#define BOTFRAMESODOMETRY_H_

#include <iostream>

#include <Eigen/Geometry>
#include <bot_frames/bot_frames.h>

class BotFramesOdometry
{
    public:
        BotFramesOdometry(BotFrames * botframes_, std::string cameraFrame);

        virtual ~BotFramesOdometry();

        void getIncrementalTransformation(Eigen::Vector3f & trans,
                                          Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                          uint64_t timestamp);
        Eigen::MatrixXd getCovariance();

        void reset();

    private:
        BotFrames* botFrames;
        Eigen::Isometry3f prevPosition; // world position of the camera at the previous iteration
        std::string cameraFrame;
        bool isInitialised; // has prevPosition been initialized?
};

#endif /* BOTFRAMESODOMETRY_H_ */

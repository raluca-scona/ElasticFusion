#ifndef BOTFRAMESODOMETRY_H_
#define BOTFRAMESODOMETRY_H_

#include <iostream>

#include <Eigen/Geometry>
#include <bot_frames/bot_frames.h>

class BotFramesOdometry
{
    public:
        BotFramesOdometry(BotFrames * botframes_, bool useVicon, std::string cameraFrame, std::string worldFrame, std::string pelvisFrame, std::string viconFrame);

        virtual ~BotFramesOdometry();

        void initialisePose(Eigen::Matrix4f & pose, uint64_t timestamp);

        void resetStartPose(uint64_t timestamp);

        void getIncrementalTransformation(Eigen::Matrix4f & deltaMotion,
                                          uint64_t timestamp);
        Eigen::MatrixXd getCovariance();

        void reset();

    private:
        BotFrames* botFrames;
        Eigen::Isometry3f currPosition = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f prevPosition = Eigen::Isometry3f::Identity();
        bool useVicon;
        std::string cameraFrame;
        std::string worldFrame;
        std::string pelvisFrame;
        std::string viconFrame;
};

#endif /* BOTFRAMESODOMETRY_H_ */

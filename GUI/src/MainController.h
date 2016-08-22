/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef MAINCONTROLLER_H_
#define MAINCONTROLLER_H_

#include <ElasticFusion.h>
#include <Utils/Parse.h>

#include "Tools/GUI.h"
#include "Tools/GroundTruthOdometry.h"
#include "Tools/RawLogReader.h"
#include "Tools/LiveLogReader.h"

// uncomment to enable benchmarking
//#define BENCHMARKCTRL

#ifdef BENCHMARKCTRL
#include <chrono>
#include <iostream>
#include <fstream>
#endif

class MainController
{
    public:
        MainController(int argc, char * argv[]);
        virtual ~MainController();

        void launch();

    private:
        void run();

        void loadCalibration(const std::string & filename);

        bool good;
        ElasticFusion * eFusion;
        GUI * gui;
        GroundTruthOdometry * groundTruthOdometry;
        LogReader * logReader;

        bool iclnuim;
        std::string logFile;
        std::string poseFile;

        float confidence,
              depth,
              icp,
              icpErrThresh,
              covThresh,
              photoThresh,
              fernThresh;

        int timeDelta,
            icpCountThresh,
            start,
            end;

        bool fillIn,
             openLoop,
             reloc,
             frameskip,
             quiet,
             fastOdom,
             so3,
             rewind,
             frameToFrameRGB;

        int framesToSkip;
        bool streaming;
        bool resetButton;

        Resize * resizeStream;

#ifdef BENCHMARKCTRL
        std::ofstream out;

        std::chrono::duration<double, std::milli> processFrameDuration;
        std::chrono::duration<double, std::milli> guiDuration;
        std::chrono::duration<double, std::milli> mainControllerDuration;

        int prevLastCount = 0;
        int currLastCount = 0;

        int prevLocalDeforms = 0;
        int currLocalDeforms = 0;

        int prevGlobalDeforms = 0;
        int currGlobalDeforms = 0;
#endif /* BENCHMARKCTRL */
};

#endif /* MAINCONTROLLER_H_ */

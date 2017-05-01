#ifndef IMAGESLOGREADER_H
#define IMAGESLOGREADER_H

#include <Utils/Resolution.h>
#include <Utils/Stopwatch.h>
#include <pangolin/utils/file_utils.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "LogReader.h"

#include <cassert>
#include <zlib.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <stack>
#include <fstream>

//for reading files from a dir
#include <dirent.h>
#include <sys/stat.h>

class ImagesLogReader : public LogReader
{
    public:
        ImagesLogReader(std::string file, bool flipColors);

        virtual ~ImagesLogReader();

        void getNext();

        void getBack();

        int getNumFrames();

        bool hasMore();

        bool rewound();

        void rewind();

        void fastForward(int frame);

        const std::string getFile();

        void setAuto(bool value);

        std::stack<int> filePointers;

    private:
        std::vector<std::string> colourImagesNames;
        std::vector<std::string> depthImagesNames;

        std::string colourImagesDirectory;
        std::string depthImagesDirectory;

        void getCore();
};


#endif // IMAGESLOGREADER_H


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

#include "ImagesLogReader.h"

ImagesLogReader::ImagesLogReader(std::string file, bool flipColors)
 : LogReader(file, flipColors)
{
    //file = "/usr/prakt/p025/datasets/rgbd_dataset_freiburg3_walking_halfsphere/";
    this->colourImagesDirectory = file + "/rgb/";
    this->depthImagesDirectory = file + "/depth/";

    //READING FILE NAMES
    DIR *dir;
    struct dirent *ent;
    struct stat filestat;

    if ( (dir = opendir(this->colourImagesDirectory.c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
          std::string filePath = this->colourImagesDirectory + ent->d_name;

          if (stat( filePath.c_str(), &filestat )) continue;
          if (S_ISDIR( filestat.st_mode ))    continue;

          this->colourImagesNames.push_back(ent->d_name);
      }
      closedir (dir);
    } else {
      std::cout<<"\nCould not open RGB directory\n";
      exit(-1);
     }

    if ( (dir = opendir(this->depthImagesDirectory.c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
          std::string filePath = this->depthImagesDirectory + ent->d_name;

          if (stat( filePath.c_str(), &filestat )) continue;
          if (S_ISDIR( filestat.st_mode ))    continue;

          this->depthImagesNames.push_back(ent->d_name);
      }
      closedir (dir);
    } else {
      std::cout<<"\nCould not open DEPTH directory\n";
      exit(-1);
    }

    this->currentFrame = 0;
    this->numFrames = std::min(this->colourImagesNames.size(), this->depthImagesNames.size());

    std::sort(this->colourImagesNames.begin(), this->colourImagesNames.end());
    std::sort(this->depthImagesNames.begin(), this->depthImagesNames.end());

}

ImagesLogReader::~ImagesLogReader()
{
}

void ImagesLogReader::getBack()
{
    this->currentFrame = 0;

    getCore();
}

void ImagesLogReader::getNext()
{
    getCore();
}

void ImagesLogReader::getCore()
{
    std::string currentColourImageName = this->colourImagesDirectory + this->colourImagesNames[this->currentFrame];
    std::string currentDepthImageName = this->depthImagesDirectory + this->depthImagesNames[this->currentFrame];

    cv::Mat colorImage = cv::imread(currentColourImageName, CV_LOAD_IMAGE_COLOR);
    cv::Mat depthImage = cv::imread(currentDepthImageName, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH );
    cv::Mat depthImageUshort;
    depthImage.convertTo(depthImageUshort, CV_16U, 1000.0 / 5000.0);

    rgb = colorImage.data;
    depth = (unsigned short *) depthImageUshort.data;


    if(!flipColors)
    {
        for(int i = 0; i < Resolution::getInstance().numPixels() * 3; i += 3)
        {
            std::swap(rgb[i + 0], rgb[i + 2]);
        }
    }

    this->currentFrame++;

}

void ImagesLogReader::fastForward(int frame)
{
    while(currentFrame < frame && hasMore())
    {
        currentFrame++;
    }
}

int ImagesLogReader::getNumFrames()
{
    return numFrames;
}

bool ImagesLogReader::hasMore()
{
    return currentFrame < numFrames;
}


void ImagesLogReader::rewind()
{
    this->currentFrame = 0;
}

bool ImagesLogReader::rewound()
{
    return currentFrame == 0;
}

const std::string ImagesLogReader::getFile()
{
    return file;
}

void ImagesLogReader::setAuto(bool value)
{

}

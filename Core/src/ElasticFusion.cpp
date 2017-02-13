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
 
#include "ElasticFusion.h"

ElasticFusion::ElasticFusion(const int timeDelta,
                             const int countThresh,
                             const float errThresh,
                             const float covThresh,
                             const bool closeLoops,
                             const bool iclnuim,
                             const bool reloc,
                             const float photoThresh,
                             const float confidence,
                             const float depthCut,
                             const float icpThresh,
                             const bool fastOdom,
                             const float fernThresh,
                             const bool so3,
                             const bool frameToFrameRGB,
                             const std::string fileName)
 : frameToModel(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                Intrinsics::getInstance().cx(),
                Intrinsics::getInstance().cy(),
                Intrinsics::getInstance().fx(),
                Intrinsics::getInstance().fy()),
   modelToModel(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                Intrinsics::getInstance().cx(),
                Intrinsics::getInstance().cy(),
                Intrinsics::getInstance().fx(),
                Intrinsics::getInstance().fy()),
   ferns(500, depthCut * 1000, photoThresh),
   saveFilename(fileName),
   currPose(Eigen::Matrix4f::Identity()),
   tick(1),
   timeDelta(timeDelta),
   icpCountThresh(countThresh),
   icpErrThresh(errThresh),
   covThresh(covThresh),
   deforms(0),
   fernDeforms(0),
   consSample(16),
   //consSample(20),
   resize(Resolution::getInstance().width(),
          Resolution::getInstance().height(),
          Resolution::getInstance().width() / consSample,
          Resolution::getInstance().height() / consSample),
   imageBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
   consBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
   timesBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
   closeLoops(closeLoops),
   iclnuim(iclnuim),
   reloc(reloc),
   lost(false),
   lastFrameRecovery(false),
   trackingCount(0),
   maxDepthProcessed(20.0f),
   rgbOnly(false),
   icpWeight(icpThresh),
   pyramid(true),
   fastOdom(fastOdom),
   confidenceThreshold(confidence),
   fernThresh(fernThresh),
   so3(so3),
   frameToFrameRGB(frameToFrameRGB),
   depthCutoff(depthCut)
{
    createTextures();
    createCompute();
    createFeedbackBuffers();

    Stopwatch::getInstance().setCustomSignature(12431231);
}

ElasticFusion::ElasticFusion(BotFrames * botFrames,
                             const bool useBotFramesOdometry,
                             std::string cameraFrame,
                             std::string worldFrame,
                             bool useVicon,
                             std::string pelvisFrame,
                             std::string viconFrame,
                             const int timeDelta,
                             const int countThresh,
                             const float errThresh,
                             const float covThresh,
                             const bool closeLoops,
                             const bool iclnuim,
                             const bool reloc,
                             const float photoThresh,
                             const float confidence,
                             const float depthCut,
                             const float icpThresh,
                             const bool fastOdom,
                             const float fernThresh,
                             const bool so3,
                             const bool frameToFrameRGB,
                             const std::string fileName)
    : ElasticFusion(timeDelta,
                    countThresh,
                    errThresh,
                    covThresh,
                    closeLoops,
                    iclnuim,
                    reloc,
                    photoThresh,
                    confidence,
                    depthCut,
                    icpThresh,
                    fastOdom,
                    fernThresh,
                    so3,
                    frameToFrameRGB,
                    fileName)
 {
    botFrames_ = botFrames;
    botFramesOdometry = new BotFramesOdometry(botFrames, useVicon, cameraFrame, worldFrame, pelvisFrame, viconFrame);
    configCameraFrameName = cameraFrame;
    configWorldFrameName = worldFrame;

    this->useBotFramesOdometry = useBotFramesOdometry;
}



ElasticFusion::~ElasticFusion()
{

    savePlyAndTrajectory();

    for(std::map<std::string, GPUTexture*>::iterator it = textures.begin(); it != textures.end(); ++it)
    {
        delete it->second;
    }

    textures.clear();

    for(std::map<std::string, ComputePack*>::iterator it = computePacks.begin(); it != computePacks.end(); ++it)
    {
        delete it->second;
    }

    computePacks.clear();

    for(std::map<std::string, FeedbackBuffer*>::iterator it = feedbackBuffers.begin(); it != feedbackBuffers.end(); ++it)
    {
        delete it->second;
    }

    feedbackBuffers.clear();
}

void ElasticFusion::createTextures()
{
    textures[GPUTexture::RGB] = new GPUTexture(Resolution::getInstance().width(),
                                               Resolution::getInstance().height(),
                                               GL_RGBA,
                                               GL_RGB,
                                               GL_UNSIGNED_BYTE,
                                               true,
                                               true);

    textures[GPUTexture::DEPTH_RAW] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_LUMINANCE16UI_EXT,
                                                     GL_LUMINANCE_INTEGER_EXT,
                                                     GL_UNSIGNED_SHORT);

    textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                          Resolution::getInstance().height(),
                                                          GL_LUMINANCE16UI_EXT,
                                                          GL_LUMINANCE_INTEGER_EXT,
                                                          GL_UNSIGNED_SHORT,
                                                          false,
                                                          true);

    textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(Resolution::getInstance().width(),
                                                        Resolution::getInstance().height(),
                                                        GL_LUMINANCE32F_ARB,
                                                        GL_LUMINANCE,
                                                        GL_FLOAT);

    textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                                 Resolution::getInstance().height(),
                                                                 GL_LUMINANCE32F_ARB,
                                                                 GL_LUMINANCE,
                                                                 GL_FLOAT);

    textures[GPUTexture::DEPTH_NORM] = new GPUTexture(Resolution::getInstance().width(),
                                                      Resolution::getInstance().height(),
                                                      GL_LUMINANCE,
                                                      GL_LUMINANCE,
                                                      GL_FLOAT,
                                                      true);
}

void ElasticFusion::createCompute()
{
    computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"),
                                                      textures[GPUTexture::DEPTH_NORM]->texture);

    computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"),
                                                        textures[GPUTexture::DEPTH_FILTERED]->texture);

    computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
                                                        textures[GPUTexture::DEPTH_METRIC]->texture);

    computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric.frag", "quad.geom"),
                                                                 textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);
}

void ElasticFusion::createFeedbackBuffers()
{
    feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback.vert", "vertex_feedback.geom"));
}

void ElasticFusion::computeFeedbackBuffers()
{
    TICK("feedbackBuffers");
    feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture,
                                                  textures[GPUTexture::DEPTH_METRIC]->texture,
                                                  tick,
                                                  maxDepthProcessed);

    feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture,
                                                       textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture,
                                                       tick,
                                                       maxDepthProcessed);
    TOCK("feedbackBuffers");
}

bool ElasticFusion::denseEnough(const Img<Eigen::Matrix<unsigned char, 3, 1>> & img)
{
    int sum = 0;

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            sum += img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(0) > 0 &&
                   img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(1) > 0 &&
                   img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(2) > 0;
        }
    }

    return float(sum) / float(img.rows * img.cols) > 0.75f;
}

void ElasticFusion::processFrame(const unsigned char * rgb,
                                 const unsigned short * depth,
                                 std::vector<float> & icpRes,
                                 const int64_t & timestamp,
                                 const Eigen::Matrix4f * inPose,
                                 const float weightMultiplier,
                                 const bool bootstrap)
{
    TICK("Run");

    textures[GPUTexture::DEPTH_RAW]->texture->Upload(depth, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
    textures[GPUTexture::RGB]->texture->Upload(rgb, GL_RGB, GL_UNSIGNED_BYTE);

    TICK("Preprocess");

    filterDepth();
    metriciseDepth();

    TOCK("Preprocess");

    //First run
    if(tick == 1)
    {
        botFramesOdometry->initialisePose(currPose, timestamp);

        //sometimes botframes does not initialise the pose properly. if z is smaller than 1.5 meters, return
        if (currPose(2, 3) < 1.5) return;

        computeFeedbackBuffers();

        globalModel.initialise(*feedbackBuffers[FeedbackBuffer::RAW], *feedbackBuffers[FeedbackBuffer::FILTERED]);

        frameToModel.initFirstRGB(textures[GPUTexture::RGB]);
    }
    else
    {
        Eigen::Matrix4f lastPose = currPose;

        bool trackingOk = true;

        if( bootstrap || !inPose )
        {

            TICK("autoFill");
            resize.image(indexMap.imageTex(), imageBuff);
            bool shouldFillIn = !denseEnough(imageBuff);
            TOCK("autoFill");

            TICK("odomInit");
            //WARNING initICP* must be called before initRGB*
            frameToModel.initICPModel(shouldFillIn ? &fillIn.vertexTexture : indexMap.vertexTex(),
                                      shouldFillIn ? &fillIn.normalTexture : indexMap.normalTex(),
                                      maxDepthProcessed, currPose);
            frameToModel.initRGBModel((shouldFillIn || frameToFrameRGB) ? &fillIn.imageTexture : indexMap.imageTex());

            frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
            frameToModel.initRGB(textures[GPUTexture::RGB]);
            TOCK("odomInit");

            if(bootstrap)
            {
                assert(inPose);
                currPose = currPose * (*inPose);
            }

            Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

            TICK("odom");

#ifdef BENCHMARKEF
            std::chrono::high_resolution_clock::time_point odomT0  = std::chrono::high_resolution_clock::now();
#endif


            Eigen::Matrix4f deltaMotion = Eigen::Matrix4f::Identity();

            botFramesOdometry->getIncrementalTransformation(deltaMotion, timestamp);

            std::vector<float> icpResiduals (Resolution::getInstance().width() * Resolution::getInstance().height());

            frameToModel.getIncrementalTransformation(trans,
                                                      rot,
                                                      rgbOnly,
                                                      icpWeight,
                                                      pyramid,
                                                      fastOdom,
                                                      so3,
                                                      deltaMotion,
                                                      useBotFramesOdometry,
                                                      icpResiduals);

            std::copy(icpResiduals.begin(), icpResiduals.end(), icpRes.begin());

            TOCK("odom");

#ifdef BENCHMARKEF
            std::chrono::high_resolution_clock::time_point odomT1  = std::chrono::high_resolution_clock::now();
            odomDuration = odomT1 - odomT0;

            trackingICPError = frameToModel.lastICPError;
            trackingICPCount = frameToModel.lastICPCount;

            trackingRGBError = frameToModel.lastRGBError;
            trackingRGBCount = frameToModel.lastRGBCount;
#endif

            trackingOk = !reloc || frameToModel.lastICPError < 1e-04;



            if(reloc)
            {
                if(!lost)
                {
                    Eigen::MatrixXd covariance = frameToModel.getCovariance();

                    for(int i = 0; i < 6; i++)
                    {
                        if(covariance(i, i) > 1e-04)
                        {
                            trackingOk = false;
                            break;
                        }
                    }

                    if(!trackingOk)
                    {
                        trackingCount++;

                        if(trackingCount > 10)
                        {
                            lost = true;
                        }
                    }
                    else
                    {
                        trackingCount = 0;
                    }
                }
                else if(lastFrameRecovery)
                {
                    Eigen::MatrixXd covariance = frameToModel.getCovariance();

                    for(int i = 0; i < 6; i++)
                    {
                        if(covariance(i, i) > 1e-04)
                        {
                            trackingOk = false;
                            break;
                        }
                    }

                    if(trackingOk)
                    {
                        lost = false;
                        trackingCount = 0;
                    }

                    lastFrameRecovery = false;
                }
            }

            currPose.topRightCorner(3, 1) = trans;
            currPose.topLeftCorner(3, 3) = rot;
        }
        else if ( false /*useBotFramesOdometry*/ ) {

            Eigen::Matrix4f deltaMotion = Eigen::Matrix4f::Identity();

            botFramesOdometry->getIncrementalTransformation(deltaMotion, timestamp);

            currPose = currPose * deltaMotion;

        } else {
            currPose = *inPose;
        }

        Eigen::Matrix4f diff = currPose.inverse() * lastPose;

        Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
        Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

        //Weight by velocity

        float weighting = std::max(diffTrans.norm(), OdometryProvider::rodrigues2(diffRot).norm());

        float largest = 0.01;
        float minWeight = 0.5;

        if(weighting > largest)
        {
            weighting = largest;
        }

        weighting = std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

        std::vector<Ferns::SurfaceConstraint> constraints;

        predict();

        Eigen::Matrix4f recoveryPose = currPose;

#ifdef BENCHMARKEF
        std::chrono::high_resolution_clock::time_point globalClosureT0  = std::chrono::high_resolution_clock::now();
#endif

        //Done with the tracking - attempting to close global loop closure using fern database
        if(closeLoops)
        {
            lastFrameRecovery = false;


            //this adds constraints only if the registration is close enough -
            //passing in here the texture produced by predict() - using active part of model
            TICK("Ferns::findFrame");
            recoveryPose = ferns.findFrame(constraints,
                                           currPose,
                                           &fillIn.vertexTexture,
                                           &fillIn.normalTexture,
                                           &fillIn.imageTexture,
                                           tick,
                                           lost);
            TOCK("Ferns::findFrame");
        }

        std::vector<float> rawGraph;

        bool fernAccepted = false;

        //if I found a similar enough fern

        if(closeLoops && ferns.lastClosest != -1)
        {
            if(lost)
            {
                currPose = recoveryPose;
                lastFrameRecovery = true;
            }

            //if I am not lost, add the identified global constraints
            else
            {
                for(size_t i = 0; i < constraints.size(); i++)
                {
                    globalDeformation.addConstraint(constraints.at(i).sourcePoint,
                                                    constraints.at(i).targetPoint,
                                                    tick,
                                                    ferns.frames.at(ferns.lastClosest)->srcTime,
                                                    true);
                }

                //where to the relative constraints come from?
                for(size_t i = 0; i < relativeCons.size(); i++)
                {
                    globalDeformation.addConstraint(relativeCons.at(i));
                }

                //Applying the deformation graph?
                if(globalDeformation.constrain(ferns.frames, rawGraph, tick, true, poseGraph, true))
                {

                    std::cout<<"deforming the graph now - global\n";
                    currPose = recoveryPose;

                    poseMatches.push_back(PoseMatch(ferns.lastClosest, ferns.frames.size(), ferns.frames.at(ferns.lastClosest)->pose, currPose, constraints, true));

                    fernDeforms += rawGraph.size() > 0;

                    fernAccepted = true;
                }
            }
        }
#ifdef BENCHMARKEF
        std::chrono::high_resolution_clock::time_point globalClosureT1  = std::chrono::high_resolution_clock::now();
        globalClosureDuration = globalClosureT1 - globalClosureT0;
#endif

#ifdef BENCHMARKEF
        std::chrono::high_resolution_clock::time_point localClosureT0  = std::chrono::high_resolution_clock::now();
#endif
        //If we didn't match to a fern
        if(!lost && closeLoops && rawGraph.size() == 0)
        {
            //Only predict old view, since we just predicted the current view for the ferns (which failed!)

            //Using the inactive view of the IndexMap

            //Trying to register active model into inactive
            TICK("IndexMap::INACTIVE");
            indexMap.combinedPredict(currPose,
                                     globalModel.model(),
                                     maxDepthProcessed,
                                     confidenceThreshold,
                                     0,
                                     tick - timeDelta,
                                     timeDelta,
                                     IndexMap::INACTIVE);
            TOCK("IndexMap::INACTIVE");

            //WARNING initICP* must be called before initRGB*
            modelToModel.initICPModel(indexMap.oldVertexTex(), indexMap.oldNormalTex(), maxDepthProcessed, currPose);
            modelToModel.initRGBModel(indexMap.oldImageTex());

            modelToModel.initICP(indexMap.vertexTex(), indexMap.normalTex(), maxDepthProcessed);
            modelToModel.initRGB(indexMap.imageTex());

            Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

            //looking for local loop closures
            modelToModel.getIncrementalTransformation(trans,
                                                      rot,
                                                      false,
                                                      10,
                                                      pyramid,
                                                      fastOdom,
                                                      false);

            Eigen::MatrixXd covar = modelToModel.getCovariance();
            bool covOk = true;

            for(int i = 0; i < 6; i++)
            {
                if(covar(i, i) > covThresh)
                {
                    covOk = false;
                    break;
                }
            }

            Eigen::Matrix4f estPose = Eigen::Matrix4f::Identity();

            estPose.topRightCorner(3, 1) = trans;
            estPose.topLeftCorner(3, 3) = rot;

            //if registration was ok, add local deformation constraints
            if(covOk && modelToModel.lastICPCount > icpCountThresh && modelToModel.lastICPError < icpErrThresh)
            {
                resize.vertex(indexMap.vertexTex(), consBuff);
                resize.time(indexMap.oldTimeTex(), timesBuff);

                for(int i = 0; i < consBuff.cols; i++)
                {
                    for(int j = 0; j < consBuff.rows; j++)
                    {
                        if(consBuff.at<Eigen::Vector4f>(j, i)(2) > 0 &&
                           consBuff.at<Eigen::Vector4f>(j, i)(2) < maxDepthProcessed &&
                           timesBuff.at<unsigned short>(j, i) > 0)
                        {
                            Eigen::Vector4f worldRawPoint = currPose * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0),
                                                                                       consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                                                       consBuff.at<Eigen::Vector4f>(j, i)(2),
                                                                                       1.0f);

                            Eigen::Vector4f worldModelPoint = estPose * Eigen::Vector4f(consBuff.at<Eigen::Vector4f>(j, i)(0),
                                                                                        consBuff.at<Eigen::Vector4f>(j, i)(1),
                                                                                        consBuff.at<Eigen::Vector4f>(j, i)(2),
                                                                                        1.0f);

                            constraints.push_back(Ferns::SurfaceConstraint(worldRawPoint, worldModelPoint));

                            localDeformation.addConstraint(worldRawPoint,
                                                           worldModelPoint,
                                                           tick,
                                                           timesBuff.at<unsigned short>(j, i),
                                                           deforms == 0);
                        }
                    }
                }

                std::vector<Deformation::Constraint> newRelativeCons;

                if(localDeformation.constrain(ferns.frames, rawGraph, tick, false, poseGraph, false, &newRelativeCons))
                {
                    std::cout<<"deforming the graph now - local\n";

                    poseMatches.push_back(PoseMatch(ferns.frames.size() - 1, ferns.frames.size(), estPose, currPose, constraints, false));

                    deforms += rawGraph.size() > 0;

                    currPose = estPose;

                    for(size_t i = 0; i < newRelativeCons.size(); i += newRelativeCons.size() / 3)
                    {
                        relativeCons.push_back(newRelativeCons.at(i));
                    }
                }
            }
        }
#ifdef BENCHMARKEF
        std::chrono::high_resolution_clock::time_point localClosureT1  = std::chrono::high_resolution_clock::now();
        localClosureDuration = localClosureT1 - localClosureT0;
#endif


        //rgbOnly does only tracking (no fusion) - this loop means - if I also want fusion, tracking is ok and i am not lost
        //this bit of the code does the fusion
#ifdef BENCHMARKEF
        std::chrono::high_resolution_clock::time_point fusionT0  = std::chrono::high_resolution_clock::now();
#endif


        if(!rgbOnly && trackingOk && !lost )
        {
            TICK("indexMap");
            indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);
            TOCK("indexMap");

            globalModel.fuse(currPose,
                             tick,
                             textures[GPUTexture::RGB],
                             textures[GPUTexture::DEPTH_METRIC],
                             textures[GPUTexture::DEPTH_METRIC_FILTERED],
                             indexMap.indexTex(),
                             indexMap.vertConfTex(),
                             indexMap.colorTimeTex(),
                             indexMap.normalRadTex(),
                             maxDepthProcessed,
                             confidenceThreshold,
                             weighting);

            TICK("indexMap");
            indexMap.predictIndices(currPose, tick, globalModel.model(), maxDepthProcessed, timeDelta);
            TOCK("indexMap");

            //If we're deforming we need to predict the depth again to figure out which
            //points to update the timestamp's of, since a deformation means a second pose update
            //this loop
            if(rawGraph.size() > 0 && !fernAccepted)
            {
                indexMap.synthesizeDepth(currPose,
                                         globalModel.model(),
                                         maxDepthProcessed,
                                         confidenceThreshold,
                                         tick,
                                         tick - timeDelta,
                                         std::numeric_limits<unsigned short>::max());
            }

            globalModel.clean(currPose,
                              tick,
                              indexMap.indexTex(),
                              indexMap.vertConfTex(),
                              indexMap.colorTimeTex(),
                              indexMap.normalRadTex(),
                              indexMap.depthTex(),
                              confidenceThreshold,
                              rawGraph,
                              timeDelta,
                              maxDepthProcessed,
                              fernAccepted);
        }
#ifdef BENCHMARKEF
        std::chrono::high_resolution_clock::time_point fusionT1  = std::chrono::high_resolution_clock::now();
        fusionDuration = fusionT1 - fusionT0;
#endif

    }


    poseGraph.push_back(std::pair<unsigned long long int, Eigen::Matrix4f>(tick, currPose));
    poseLogTimes.push_back(timestamp);

    TICK("sampleGraph");

    localDeformation.sampleGraphModel(globalModel.model());

    globalDeformation.sampleGraphFrom(localDeformation);

    TOCK("sampleGraph");

    predict();

    if(!lost)
    {
        processFerns();
        tick++;
    }

    TOCK("Run");
}

void ElasticFusion::processFerns()
{
    TICK("Ferns::addFrame");
    ferns.addFrame(&fillIn.imageTexture, &fillIn.vertexTexture, &fillIn.normalTexture, currPose, tick, fernThresh);
    TOCK("Ferns::addFrame");
}

void ElasticFusion::predict()
{
    TICK("IndexMap::ACTIVE");

    if(lastFrameRecovery)
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 0,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }
    else
    {
        indexMap.combinedPredict(currPose,
                                 globalModel.model(),
                                 maxDepthProcessed,
                                 confidenceThreshold,
                                 tick,
                                 tick,
                                 timeDelta,
                                 IndexMap::ACTIVE);
    }

    TICK("FillIn");
    fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], lost);
    fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], lost);
    fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], lost || frameToFrameRGB);
    TOCK("FillIn");

    TOCK("IndexMap::ACTIVE");
}

void ElasticFusion::metriciseDepth()
{
    std::vector<Uniform> uniforms;

    uniforms.push_back(Uniform("maxD", depthCutoff));

    computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
    computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
}

void ElasticFusion::filterDepth()
{
    std::vector<Uniform> uniforms;

    uniforms.push_back(Uniform("cols", (float)Resolution::getInstance().cols()));
    uniforms.push_back(Uniform("rows", (float)Resolution::getInstance().rows()));
    uniforms.push_back(Uniform("maxD", depthCutoff));

    computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void ElasticFusion::normaliseDepth(const float & minVal, const float & maxVal)
{
    std::vector<Uniform> uniforms;

    uniforms.push_back(Uniform("maxVal", maxVal * 1000.f));
    uniforms.push_back(Uniform("minVal", minVal * 1000.f));

    computePacks[ComputePack::NORM]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void ElasticFusion::savePlyAndTrajectory()
{
    std::string filename = "/home/raluca/ef-statistics/ef-map-"+saveFilename+".ply";

    // Open file
    std::ofstream fs;
    fs.open (filename.c_str ());

    Eigen::Vector4f * mapData = globalModel.downloadMap();

    int validCount = 0;

    for(unsigned int i = 0; i < globalModel.lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 3) + 0];

        if(pos[3] > confidenceThreshold)
        {
            validCount++;
        }
    }

    // Write header
    fs << "ply";
    fs << "\nformat " << "binary_little_endian" << " 1.0";

    // Vertices
    fs << "\nelement vertex "<< validCount;
    fs << "\nproperty float x"
          "\nproperty float y"
          "\nproperty float z";

    fs << "\nproperty uchar red"
          "\nproperty uchar green"
          "\nproperty uchar blue";

    fs << "\nproperty float nx"
          "\nproperty float ny"
          "\nproperty float nz";

    fs << "\nproperty float radius";

    fs << "\nend_header\n";

    // Close the file
    fs.close ();

    // Open file in binary appendable
    std::ofstream fpout (filename.c_str (), std::ios::app | std::ios::binary);

    for(unsigned int i = 0; i < globalModel.lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 3) + 0];

        if(pos[3] > confidenceThreshold)
        {
            Eigen::Vector4f col = mapData[(i * 3) + 1];
            Eigen::Vector4f nor = mapData[(i * 3) + 2];

            nor[0] *= -1;
            nor[1] *= -1;
            nor[2] *= -1;

            float value;
            memcpy (&value, &pos[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            unsigned char r = int(col[0]) >> 16 & 0xFF;
            unsigned char g = int(col[0]) >> 8 & 0xFF;
            unsigned char b = int(col[0]) & 0xFF;

            fpout.write (reinterpret_cast<const char*> (&r), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&g), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&b), sizeof (unsigned char));

            memcpy (&value, &nor[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[3], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));
        }
    }

    // Close file
    fs.close ();

    delete [] mapData;

    //Output deformed pose graph
    std::string fname =  "/home/raluca/ef-statistics/" + saveFilename + "-ef-trajectory.txt";

    std::ofstream f;
    f.open(fname.c_str(), std::fstream::out);

    for(size_t i = 0; i < poseGraph.size(); i++)
    {
        std::stringstream strs;

        if(iclnuim)
        {
            strs << std::setprecision(6) << std::fixed << (double)poseLogTimes.at(i) << " ";
        }
        else
        {
            strs << std::setprecision(6) << std::fixed << (double)poseLogTimes.at(i) /*/ 1000000.0 */<< " ";
        }

        Eigen::Matrix4f headToNominalCamera = Eigen::Matrix4f::Zero();
        Eigen::Matrix4f cameraToHead = Eigen::Matrix4f::Zero();
        Eigen::Matrix4f cameraRightHand = Eigen::Matrix4f::Zero();
        headToNominalCamera(0, 0) = 1;
        headToNominalCamera(1, 1) = -1;
        headToNominalCamera(1, 3) =  0.0350;
        headToNominalCamera(2, 2) = -1;
        headToNominalCamera(2, 3) = -0.002;
        headToNominalCamera(3, 3) = 1;

        cameraToHead(0, 1) = -1;
        cameraToHead(0, 3) = 0.035;
        cameraToHead(1, 2) = -1;
        cameraToHead(1, 3) = -0.002;
        cameraToHead(2, 0) = 1;
        cameraToHead(3, 3) = 1;

        cameraRightHand = cameraToHead * headToNominalCamera;

        Eigen::Matrix4f poseInRightHandSystem = poseGraph.at(i).second * cameraRightHand;

        Eigen::Vector3f trans = poseInRightHandSystem.topRightCorner(3, 1);
        Eigen::Matrix3f rot = poseInRightHandSystem.topLeftCorner(3, 3);

        f << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

        Eigen::Quaternionf currentCameraRotation(rot);

        f << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";
    }

    f.close();

}

//Sad times ahead
IndexMap & ElasticFusion::getIndexMap()
{
    return indexMap;
}

GlobalModel & ElasticFusion::getGlobalModel()
{
    return globalModel;
}

Ferns & ElasticFusion::getFerns()
{
    return ferns;
}

Deformation & ElasticFusion::getLocalDeformation()
{
    return localDeformation;
}

std::map<std::string, GPUTexture*> & ElasticFusion::getTextures()
{
    return textures;
}

const std::vector<PoseMatch> & ElasticFusion::getPoseMatches()
{
    return poseMatches;
}

const RGBDOdometry & ElasticFusion::getModelToModel()
{
    return modelToModel;
}

const float & ElasticFusion::getConfidenceThreshold()
{
    return confidenceThreshold;
}

void ElasticFusion::setRgbOnly(const bool & val)
{
    rgbOnly = val;
}

void ElasticFusion::setIcpWeight(const float & val)
{
    icpWeight = val;
}

void ElasticFusion::setPyramid(const bool & val)
{
    pyramid = val;
}

void ElasticFusion::setFastOdom(const bool & val)
{
    fastOdom = val;
}

void ElasticFusion::setSo3(const bool & val)
{
    so3 = val;
}

void ElasticFusion::setFrameToFrameRGB(const bool & val)
{
    frameToFrameRGB = val;
}

void ElasticFusion::setConfidenceThreshold(const float & val)
{
    confidenceThreshold = val;
}

void ElasticFusion::setFernThresh(const float & val)
{
    fernThresh = val;
}

void ElasticFusion::setDepthCutoff(const float & val)
{
    depthCutoff = val;
}

const bool & ElasticFusion::getLost() //lel
{
    return lost;
}

const int & ElasticFusion::getTick()
{
    return tick;
}

const int & ElasticFusion::getTimeDelta()
{
    return timeDelta;
}

void ElasticFusion::setTick(const int & val)
{
    tick = val;
}

const float & ElasticFusion::getMaxDepthProcessed()
{
    return maxDepthProcessed;
}

const Eigen::Matrix4f & ElasticFusion::getCurrPose()
{
    return currPose;
}

const int & ElasticFusion::getDeforms()
{
    return deforms;
}

const int & ElasticFusion::getFernDeforms()
{
    return fernDeforms;
}

std::map<std::string, FeedbackBuffer*> & ElasticFusion::getFeedbackBuffers()
{
    return feedbackBuffers;
}

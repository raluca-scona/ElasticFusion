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

#include "RGBDOdometry.h"

RGBDOdometry::RGBDOdometry(int width,
                           int height,
                           float cx, float cy, float fx, float fy,
                           float distThresh,
                           float angleThresh)
: lastICPError(0),
  lastICPCount(width * height),
  lastRGBError(0),
  lastRGBCount(width * height),
  lastSO3Error(0),
  lastSO3Count(width * height),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  lastb(Eigen::Matrix<double, 6, 1>::Zero()),
  sobelSize(3),
  sobelScale(1.0 / pow(2.0, sobelSize)),
  maxDepthDeltaRGB(0.07),
  maxDepthRGB(6.0),
  distThres_(distThresh),
  angleThres_(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy)
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);
    sumResidualRGB.create(MAX_THREADS);

    sumDataSO3.create(MAX_THREADS);
    outDataSO3.create(1);

    for(int i = 0; i < NUM_PYRS; i++)
    {
        int2 nextDim = {height >> i, width >> i};
        pyrDims.push_back(nextDim);
    }

    for(int i = 0; i < NUM_PYRS; i++)
    {
        lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    }

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.reserve(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);
    nmaps_g_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

        vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    }

    vmaps_tmp.create(height * 4 * width);
    nmaps_tmp.create(height * 4 * width);

    minimumGradientMagnitudes.reserve(NUM_PYRS);
    minimumGradientMagnitudes[0] = 5;
    minimumGradientMagnitudes[1] = 3;
    minimumGradientMagnitudes[2] = 1;
}

RGBDOdometry::~RGBDOdometry()
{

}

void RGBDOdometry::initICP(GPUTexture * filteredDepth, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &filteredDepth->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepth->cudaRes, 0, 0);

    cudaMemcpy2DFromArray(depth_tmp[0].ptr(0), depth_tmp[0].step(), textPtr, 0, 0, depth_tmp[0].colsBytes(), depth_tmp[0].rows(), cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &filteredDepth->cudaRes);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff);
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICP(GPUTexture * predictedVertices, GPUTexture * predictedNormals, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
        resizeNMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICPModel(GPUTexture * predictedVertices,
                                GPUTexture * predictedNormals,
                                const float depthCutoff,
                                const Eigen::Matrix4f & modelPose)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
        resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
    }

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    mat33 device_Rcam = Rcam;
    float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData(GPUTexture * rgb,
                                    DeviceArray2D<float> * destDepths,
                                    DeviceArray2D<unsigned char> * destImages)
{
    verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownGaussF(destDepths[i], destDepths[i + 1]);
    }

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, destImages[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
    populateRGBDData(rgb, &lastDepth[0], &lastImage[0]);
}

void RGBDOdometry::initRGB(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
    populateRGBDData(rgb, &nextDepth[0], &nextImage[0]);
}

void RGBDOdometry::initFirstRGB(GPUTexture * rgb)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, lastNextImage[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
    }
}

void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                const bool & rgbOnly,
                                                const float & icpWeight,
                                                const bool & pyramid,
                                                const bool & fastOdom,
                                                const bool & so3)
{
    bool icp = !rgbOnly && icpWeight > 0;
    bool rgb = rgbOnly || icpWeight < 100;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    if(rgb)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
        }
    }

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    if(so3)
    {
        int pyramidLevel = 2;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(pyramidLevel).fx;
        K(1, 1) = intr(pyramidLevel).fy;
        K(0, 2) = intr(pyramidLevel).cx;
        K(1, 2) = intr(pyramidLevel).cy;
        K(2, 2) = 1;

        float lastError = std::numeric_limits<float>::max() / 2;
        float lastCount = std::numeric_limits<float>::max() / 2;

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

        for(int i = 0; i < 10; i++)
        {
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
            Eigen::Matrix<float, 3, 1> jtr;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

            mat33 imageBasis;
            memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
            mat33 kinv;
            memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
            mat33 krlr;
            memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));

            float residual[2];

            TICK("so3Step");
            so3Step(lastNextImage[pyramidLevel],
                    nextImage[pyramidLevel],
                    imageBasis,
                    kinv,
                    krlr,
                    sumDataSO3,
                    outDataSO3,
                    jtj.data(),
                    jtr.data(),
                    &residual[0],
                    GPUConfig::getInstance().so3StepThreads,
                    GPUConfig::getInstance().so3StepBlocks);
            TOCK("so3Step");

            lastSO3Error = sqrt(residual[0]) / residual[1];
            lastSO3Count = residual[1];

            //Converged
            if(lastSO3Error < lastError && fabs(lastError - lastSO3Count) < 0.001)
            {
                break;
            }
            else if(lastSO3Error > lastError + 0.001) //Diverging
            {
                lastSO3Error = lastError;
                lastSO3Count = lastCount;
                resultR = lastResultR;
                break;
            }

            lastError = lastSO3Error;
            lastCount = lastSO3Count;
            lastResultR = resultR;

            Eigen::Vector3f delta = jtj.ldlt().solve(jtr);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());

            R_lr = rotUpdate.cast<float>() * R_lr;

            for(int x = 0; x < 3; x++)
            {
                for(int y = 0; y < 3; y++)
                {
                    resultR(x, y) = R_lr(x, y);
                }
            }
        }
    }

    iterations[0] = fastOdom ? 3 : 10; //15 for pronto on short log.
    iterations[1] = pyramid ? 5 : 0; //8
    iterations[2] = pyramid ? 4 : 0; //5

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    mat33 device_Rprev_inv = Rprev_inv;
    float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    if(so3)
    {
        for(int x = 0; x < 3; x++)
        {
            for(int y = 0; y < 3; y++)
            {
                resultRt(x, y) = resultR(x, y);
            }
        }
    }

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        if(rgb)
        {
            projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
        }

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(i).fx;
        K(1, 1) = intr(i).fy;
        K(0, 2) = intr(i).cx;
        K(1, 2) = intr(i).cy;
        K(2, 2) = 1;

        lastRGBError = std::numeric_limits<float>::max();

        for(int j = 0; j < iterations[i]; j++)
        {
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
            mat33 krkInv;
            memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
            Kt = K * Kt;
            float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

            int sigma = 0;
            int rgbSize = 0;

            if(rgb)
            {
                TICK("computeRgbResidual");
                computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
                                   nextdIdx[i],
                                   nextdIdy[i],
                                   lastDepth[i],
                                   nextDepth[i],
                                   lastImage[i],
                                   nextImage[i],
                                   corresImg[i],
                                   sumResidualRGB,
                                   maxDepthDeltaRGB,
                                   kt,
                                   krkInv,
                                   sigma,
                                   rgbSize,
                                   GPUConfig::getInstance().rgbResThreads,
                                   GPUConfig::getInstance().rgbResBlocks);
                TOCK("computeRgbResidual");
            }

            float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);

            if(rgbOnly && sqrt(sigma) / rgbSize > lastRGBError)
            {
                break;
            }

            lastRGBError = sqrt(sigma) / rgbSize;
            lastRGBCount = rgbSize;

            if(rgbOnly)
            {
                sigmaVal = -1; //Signals the internal optimisation to weight evenly
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            mat33 device_Rcurr = Rcurr;
            float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            float residual[2];

            DeviceArray2D<float> icp_perpixel_residual;
            icp_perpixel_residual.create(vmap_curr.cols(), vmap_curr.cols());

            if(icp)
            {
                TICK("icpStep");
                icpStep(device_Rcurr,
                        device_tcurr,
                        vmap_curr,
                        nmap_curr,
                        device_Rprev_inv,
                        device_tprev,
                        intr(i),
                        vmap_g_prev,
                        nmap_g_prev,
                        distThres_,
                        angleThres_,
                        sumDataSE3,
                        outDataSE3,
                        A_icp.data(),
                        b_icp.data(),
                        &residual[0],
                        icp_perpixel_residual,
                        GPUConfig::getInstance().icpStepThreads,
                        GPUConfig::getInstance().icpStepBlocks);
                TOCK("icpStep");
            }

            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            if(rgb)
            {
                TICK("rgbStep");
                rgbStep(corresImg[i],
                        sigmaVal,
                        pointClouds[i],
                        intr(i).fx,
                        intr(i).fy,
                        nextdIdx[i],
                        nextdIdy[i],
                        sobelScale,
                        sumDataSE3,
                        outDataSE3,
                        A_rgbd.data(),
                        b_rgbd.data(),
                        GPUConfig::getInstance().rgbStepThreads,
                        GPUConfig::getInstance().rgbStepBlocks);
                TOCK("rgbStep");
            }

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();

            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            if(icp && rgb)
            {
                double w = icpWeight;
                lastA = dA_rgbd + w * w * dA_icp;
                lastb = db_rgbd + w * db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(icp)
            {
                lastA = dA_icp;
                lastb = db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(rgb)
            {
                lastA = dA_rgbd;
                lastb = db_rgbd;
                result = lastA.ldlt().solve(lastb);
            }
            else
            {
                assert(false && "Control shouldn't reach here");
            }

            Eigen::Isometry3f rgbOdom;

            OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * rgbOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    if(rgb && (tcurr - tprev).norm() > 0.3)
    {
        Rcurr = Rprev;
        tcurr = tprev;
    }

    if(so3)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            std::swap(lastNextImage[i], nextImage[i]);
        }
    }

    trans = tcurr;
    rot = Rcurr;
}


void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                const bool & rgbOnly,
                                                const float & icpWeight,
                                                const bool & pyramid,
                                                const bool & fastOdom,
                                                const bool & so3,
                                                Eigen::Matrix4f botFramesDelta,
                                                const bool useBotFramesOdometry,
                                                std::vector<float> & icpResiduals)
{

    bool icp = !rgbOnly && icpWeight > 0;
    bool rgb = rgbOnly || icpWeight < 100;
    bool bot = useBotFramesOdometry;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    if(rgb)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
        }
    }

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    if(so3)
    {
        int pyramidLevel = 2;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(pyramidLevel).fx;
        K(1, 1) = intr(pyramidLevel).fy;
        K(0, 2) = intr(pyramidLevel).cx;
        K(1, 2) = intr(pyramidLevel).cy;
        K(2, 2) = 1;

        float lastError = std::numeric_limits<float>::max() / 2;
        float lastCount = std::numeric_limits<float>::max() / 2;

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

        for(int i = 0; i < 10; i++)
        {
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
            Eigen::Matrix<float, 3, 1> jtr;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

            mat33 imageBasis;
            memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
            mat33 kinv;
            memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
            mat33 krlr;
            memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));

            float residual[2];

            TICK("so3Step");
            so3Step(lastNextImage[pyramidLevel],
                    nextImage[pyramidLevel],
                    imageBasis,
                    kinv,
                    krlr,
                    sumDataSO3,
                    outDataSO3,
                    jtj.data(),
                    jtr.data(),
                    &residual[0],
                    GPUConfig::getInstance().so3StepThreads,
                    GPUConfig::getInstance().so3StepBlocks);
            TOCK("so3Step");

            lastSO3Error = sqrt(residual[0]) / residual[1];
            lastSO3Count = residual[1];

            //Converged
            if(lastSO3Error < lastError && fabs(lastError - lastSO3Count) < 0.001)
            {
                break;
            }
            else if(lastSO3Error > lastError + 0.001) //Diverging
            {
                lastSO3Error = lastError;
                lastSO3Count = lastCount;
                resultR = lastResultR;
                break;
            }

            lastError = lastSO3Error;
            lastCount = lastSO3Count;
            lastResultR = resultR;

            Eigen::Vector3f delta = jtj.ldlt().solve(jtr);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());

            R_lr = rotUpdate.cast<float>() * R_lr;

            for(int x = 0; x < 3; x++)
            {
                for(int y = 0; y < 3; y++)
                {
                    resultR(x, y) = R_lr(x, y);
                }
            }
        }
    }

    iterations[0] = fastOdom ? 3 : 10;
    iterations[1] = pyramid ? 5 : 0;
    iterations[2] = pyramid ? 4 : 0;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    mat33 device_Rprev_inv = Rprev_inv;
    float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    if(so3)
    {

        for(int x = 0; x < 3; x++)
        {
            for(int y = 0; y < 3; y++)
            {
                resultRt(x, y) = resultR(x, y);
            }
        }
    }

    float numIters = 0.;

    for (int i=0; i<NUM_PYRS; i++) {
        numIters += iterations[i];
    }


    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        if(rgb)
        {
            projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
        }

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(i).fx;
        K(1, 1) = intr(i).fy;
        K(0, 2) = intr(i).cx;
        K(1, 2) = intr(i).cy;
        K(2, 2) = 1;

        lastRGBError = std::numeric_limits<float>::max();

        for(int j = 0; j < iterations[i]; j++)
        {
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
            mat33 krkInv;
            memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
            Kt = K * Kt;
            float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

            int sigma = 0;
            int rgbSize = 0;

            if(rgb)
            {
                TICK("computeRgbResidual");
                computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
                                   nextdIdx[i],
                                   nextdIdy[i],
                                   lastDepth[i],
                                   nextDepth[i],
                                   lastImage[i],
                                   nextImage[i],
                                   corresImg[i],
                                   sumResidualRGB,
                                   maxDepthDeltaRGB,
                                   kt,
                                   krkInv,
                                   sigma,
                                   rgbSize,
                                   GPUConfig::getInstance().rgbResThreads,
                                   GPUConfig::getInstance().rgbResBlocks);
                TOCK("computeRgbResidual");
            }

            float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);

            if(rgbOnly && sqrt(sigma) / rgbSize > lastRGBError)
            {
                break;
            }

            lastRGBError = sqrt(sigma) / rgbSize;
            lastRGBCount = rgbSize;

            if(rgbOnly)
            {
                sigmaVal = -1; //Signals the internal optimisation to weight evenly
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            mat33 device_Rcurr = Rcurr;
            float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            float residual[2];

            DeviceArray2D<float> icp_perpixel_residual;
            icp_perpixel_residual.create(vmap_curr.cols(), vmap_curr.cols());

            if(icp)
            {
                TICK("icpStep");
                icpStep(device_Rcurr,
                        device_tcurr,
                        vmap_curr,
                        nmap_curr,
                        device_Rprev_inv,
                        device_tprev,
                        intr(i),
                        vmap_g_prev,
                        nmap_g_prev,
                        distThres_,
                        angleThres_,
                        sumDataSE3,
                        outDataSE3,
                        A_icp.data(),
                        b_icp.data(),
                        &residual[0],
                        icp_perpixel_residual,
                        GPUConfig::getInstance().icpStepThreads,
                        GPUConfig::getInstance().icpStepBlocks);
                TOCK("icpStep");
            }

            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            if (j == (iterations[i] - 1) && i == 0) {
                float icpRes[ vmap_curr.cols() * vmap_curr.rows() /3];

                icp_perpixel_residual.download(&icpRes[0], icp_perpixel_residual.cols() * 4);

                std::copy(icpRes, icpRes + vmap_curr.cols() * vmap_curr.cols(), icpResiduals.begin());
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            if(rgb)
            {
                TICK("rgbStep");
                rgbStep(corresImg[i],
                        sigmaVal,
                        pointClouds[i],
                        intr(i).fx,
                        intr(i).fy,
                        nextdIdx[i],
                        nextdIdy[i],
                        sobelScale,
                        sumDataSE3,
                        outDataSE3,
                        A_rgbd.data(),
                        b_rgbd.data(),
                        GPUConfig::getInstance().rgbStepThreads,
                        GPUConfig::getInstance().rgbStepBlocks);
                TOCK("rgbStep");
            }

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();

            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            Eigen::Matrix<double, 6, 1> currRtVec;
            Eigen::Matrix<double, 6, 1> botFramesVec;
            Eigen::Matrix<double, 6, 1> botFrames6DoFResidual;


            Eigen::Vector3f rtRot = OdometryProvider::rodrigues2(resultRt.cast<float>().topLeftCorner(3,3));

            currRtVec(0, 0) = resultRt(0, 3);
            currRtVec(1, 0) = resultRt(1, 3);
            currRtVec(2, 0) = resultRt(2, 3);

            currRtVec(3, 0) = rtRot(0);
            currRtVec(4, 0) = rtRot(1);
            currRtVec(5, 0) = rtRot(2);

            Eigen::Matrix4f botInv = botFramesDelta.inverse();
            Eigen::Matrix3f botInvRot = botInv.topLeftCorner(3,3);
            Eigen::Vector3f botInvRotTwist = OdometryProvider::rodrigues2(botInvRot);


            botFramesVec(0, 0) = botInv(0, 3);
            botFramesVec(1, 0) = botInv(1, 3);
            botFramesVec(2, 0) = botInv(2, 3);

            botFramesVec(3, 0) = botInvRotTwist(0);
            botFramesVec(4, 0) = botInvRotTwist(1);
            botFramesVec(5, 0) = botInvRotTwist(2);

            /*  Approximate method - difference directly on twists
            botFrames6DoFResidual = botFramesVec - currRtVec;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> botFramesJac = Eigen::Matrix<double, 6,6, Eigen::RowMajor>::Identity();
            */

            //using Lie Algebra representation for Rotations
            Eigen::Matrix3f resultRtRot = resultRt.cast<float>().topLeftCorner(3,3);
            Eigen::Matrix3f rotResidualMatrix = botInvRot * resultRtRot.inverse();
            Eigen::Vector3f rotResidual = OdometryProvider::rodrigues2(botInvRot * resultRtRot.inverse());

            botFrames6DoFResidual(0, 0) = botFramesVec(0, 0) - currRtVec(0, 0);
            botFrames6DoFResidual(1, 0) = botFramesVec(1, 0) - currRtVec(1, 0);
            botFrames6DoFResidual(2, 0) = botFramesVec(2, 0) - currRtVec(2, 0);
            botFrames6DoFResidual(3, 0) = !isnan(rotResidual(0, 0)) ? rotResidual(0, 0) : (botFramesVec(3, 0) - currRtVec(3, 0));
            botFrames6DoFResidual(4, 0) = !isnan(rotResidual(1, 0)) ? rotResidual(1, 0) : (botFramesVec(4, 0) - currRtVec(4, 0));
            botFrames6DoFResidual(5, 0) = !isnan(rotResidual(2, 0)) ? rotResidual(2, 0) : (botFramesVec(5, 0) - currRtVec(5, 0));

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> botFramesJac = Eigen::Matrix<double, 6,6, Eigen::RowMajor>::Identity();
            botFramesJac(0, 0) = -1;
            botFramesJac(1, 1) = -1;
            botFramesJac(2, 2) = -1;

            //computing Jacobian - move this to its own class
            double r11, r12, r13, r21, r22, r23, r31, r32, r33;
            r11 = rotResidualMatrix(0, 0);
            r12 = rotResidualMatrix(0, 1);
            r13 = rotResidualMatrix(0, 2);

            r21 = rotResidualMatrix(1, 0);
            r22 = rotResidualMatrix(1, 1);
            r23 = rotResidualMatrix(1, 2);

            r31 = rotResidualMatrix(2, 0);
            r32 = rotResidualMatrix(2, 1);
            r33 = rotResidualMatrix(2, 2);

            double traceR = r11 + r22 + r33;

            double d1dx = -0.5*(1.0*r22 + 1.0*r33)*pow((-1.0* pow((0.5*traceR - 0.5), 2) + 1.0),-0.5)*acos(0.5*traceR - 0.5) - (-1.0*r23 + r32)*(0.25*r23 - 0.25*r32)/( pow((0.5*traceR - 0.5), 2) - 1.0) - 0.5*(-1.0*r23 + r32)*(0.5*r23 - 0.5*r32)*(0.5*traceR - 0.5)*pow((-1.0* pow((0.5*traceR - 0.5), 2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            double d1dy = 0.5*r21*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -0.5)*acos(0.5*traceR - 0.5) - (0.25*r13 - 0.25*r31)*(1.0*r23 - 1.0*r32)/( pow((0.5*traceR - 0.5),2) - 1.0) - 0.5*(0.5*r13 - 0.5*r31)*(1.0*r23 - 1.0*r32)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            double d1dz = 0.5*r31* pow((-1.0* pow((0.5*traceR - 0.5),2) + 1.0), -0.5)*acos(0.5*traceR - 0.5) + (0.25*r12 - 0.25*r21)*(1.0*r23 - 1.0*r32)/( pow((0.5*traceR - 0.5),2) - 1.0) + 0.5*(0.5*r12 - 0.5*r21)*(1.0*r23 - 1.0*r32)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0),-1.5)*acos(0.5*traceR - 0.5);

            double d2dx = 0.5*r12*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -0.5)*acos(0.5*traceR - 0.5) + (-1.0*r13 + 1.0*r31)*(0.25*r23 - 0.25*r32)/( pow((0.5*traceR - 0.5),2) - 1.0) + 0.5*(-1.0*r13 + 1.0*r31)*(0.5*r23 - 0.5*r32)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            double d2dy = -0.5*(1.0*r11 + 1.0*r33)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0),-0.5)*acos(0.5*traceR - 0.5) + (0.25*r13 - 0.25*r31)*(1.0*r13 - 1.0*r31)/( pow((0.5*traceR - 0.5),2) - 1.0) + 0.5*(0.5*r13 - 0.5*r31)*(1.0*r13 - 1.0*r31)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            double d2dz = 0.5*r32*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0),-0.5)*acos(0.5*traceR - 0.5) - (0.25*r12 - 0.25*r21)*(1.0*r13 - 1.0*r31)/(pow((0.5*traceR - 0.5),2) - 1.0) - 0.5*(0.5*r12 - 0.5*r21)*(1.0*r13 - 1.0*r31)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0),-1.5)*acos(0.5*traceR - 0.5);

            double d3dx = 0.5*r13*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -0.5)*acos(0.5*traceR - 0.5) + (1.0*r12 - 1.0*r21)*(0.25*r23 - 0.25*r32)/(pow((0.5*traceR - 0.5),2) - 1.0) + 0.5*(1.0*r12 - 1.0*r21)*(0.5*r23 - 0.5*r32)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            double d3dy = 0.5*r23*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -0.5)*acos(0.5*traceR - 0.5) + (-1.0*r12 + r21)*(0.25*r13 - 0.25*r31)/(pow((0.5*traceR - 0.5),2) - 1.0) + 0.5*(-1.0*r12 + r21)*(0.5*r13 - 0.5*r31)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            double d3dz = -0.5*(1.0*r11 + 1.0*r22)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0),-0.5)*acos(0.5*traceR - 0.5) - (-1.0*r12 + r21)*(0.25*r12 - 0.25*r21)/(pow((0.5*traceR - 0.5),2) - 1.0) - 0.5*(-1.0*r12 + r21)*(0.5*r12 - 0.5*r21)*(0.5*traceR - 0.5)*pow((-1.0*pow((0.5*traceR - 0.5),2) + 1.0), -1.5)*acos(0.5*traceR - 0.5);

            botFramesJac(0, 0) = -1;
            botFramesJac(1, 1) = -1;
            botFramesJac(2, 2) = -1;

            botFramesJac(3, 3) = !isnan(d1dx) ? d1dx : -1;
            botFramesJac(3, 4) = !isnan(d1dy) ? d1dy : 0;
            botFramesJac(3, 5) = !isnan(d1dz) ? d1dz : 0;

            botFramesJac(4, 3) = !isnan(d2dx) ? d2dx : 0;
            botFramesJac(4, 4) = !isnan(d2dy) ? d2dy : -1;
            botFramesJac(4, 5) = !isnan(d2dz) ? d2dz : 0;

            botFramesJac(5, 3) = !isnan(d3dx) ? d3dx : 0;
            botFramesJac(5, 4) = !isnan(d3dy) ? d3dy : 0;
            botFramesJac(5, 5) = !isnan(d3dz) ? d3dz : -1;

            double numPixels = vmap_curr.cols() * vmap_curr.rows() / 3.0;

            float icpCountPercentage = lastICPCount * 100. / numPixels;
            float rgbCountPercentage = lastRGBCount * 100. / numPixels;
            float botFramesCount = ( std::max (std::max(icpCountPercentage, rgbCountPercentage) + 10. , 15.) ) / 100. * numPixels  ;

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_bot = botFramesCount * botFramesJac.transpose() * botFramesJac;
            Eigen::Matrix<double, 6, 1> db_bot = -1 * botFramesCount * botFramesJac.transpose() * botFrames6DoFResidual;

            if(bot)
            {
                double wbot = 100.;

                lastA = wbot * dA_bot;
                lastb = wbot * db_bot;

                if (rgb) {
                    lastA += dA_rgbd;
                    lastb += db_rgbd;
                }

                if (icp) {
                    double w = icpWeight;

                    lastA += w * w * dA_icp;
                    lastb += w * db_icp;
                }

                result = lastA.ldlt().solve(lastb);
            }
            else if(icp && rgb)
            {
                double w = icpWeight;
                lastA =   dA_rgbd + w * w * dA_icp;
                lastb =  db_rgbd + w * db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(icp)
            {
                lastA = dA_icp;
                lastb = db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(rgb)
            {
                lastA = dA_rgbd;
                lastb = db_rgbd;
                result = lastA.ldlt().solve(lastb);
            }
            else
            {
                assert(false && "Control shouldn't reach here");
            }

            Eigen::Isometry3f rgbOdom;

            OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * rgbOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    if(rgb && (tcurr - tprev).norm() > 0.3)
    {
        Rcurr = Rprev;
        tcurr = tprev;
    }

    if(so3)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            std::swap(lastNextImage[i], nextImage[i]);
        }
    }

    trans = tcurr;
    rot = Rcurr;
}






Eigen::MatrixXd RGBDOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}

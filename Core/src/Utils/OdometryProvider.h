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

#ifndef ODOMETRYPROVIDER_H_
#define ODOMETRYPROVIDER_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <float.h>

class OdometryProvider
{
    public:
        OdometryProvider()
        {}

        virtual ~OdometryProvider()
        {}

        static inline Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rodrigues(const Eigen::Vector3d & src)
        {
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> dst = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

            double rx, ry, rz, theta;

            rx = src(0);
            ry = src(1);
            rz = src(2);

            theta = src.norm();

            if(theta >= DBL_EPSILON)
            {
                const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

                double c = cos(theta);
                double s = sin(theta);
                double c1 = 1. - c;
                double itheta = theta ? 1./theta : 0.;

                rx *= itheta; ry *= itheta; rz *= itheta;

                double rrt[] = { rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz };
                double _r_x_[] = { 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };
                double R[9];

                for(int k = 0; k < 9; k++)
                {
                    R[k] = c*I[k] + c1*rrt[k] + s*_r_x_[k];
                }

                memcpy(dst.data(), &R[0], sizeof(Eigen::Matrix<double, 3, 3, Eigen::RowMajor>));
            }

            return dst;
        }

        static inline void computeUpdateSE3(Eigen::Matrix<double, 4, 4, Eigen::RowMajor> & resultRt, const Eigen::Matrix<double, 6, 1> & result, Eigen::Isometry3f & rgbOdom)
        {
            // for infinitesimal transformation
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

            Eigen::Vector3d rvec(result(3), result(4), result(5));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = rodrigues(rvec);

            Rt.topLeftCorner(3, 3) = R;
            Rt(0, 3) = result(0);
            Rt(1, 3) = result(1);
            Rt(2, 3) = result(2);

            resultRt = Rt * resultRt;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotation = resultRt.topLeftCorner(3, 3);
            rgbOdom.setIdentity();
            rgbOdom.rotate(rotation.cast<float>().eval());
            rgbOdom.translation() = resultRt.cast<float>().eval().topRightCorner(3, 1);
        }


        static inline Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix) {

            Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
            Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

            double rx = R(2, 1) - R(1, 2);
            double ry = R(0, 2) - R(2, 0);
            double rz = R(1, 0) - R(0, 1);

            double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
            double c = (R.trace() - 1) * 0.5;
            c = c > 1. ? 1. : c < -1. ? -1. : c;

            double theta = acos(c);

            if( s < 1e-5 )
            {
                double t;

                if( c > 0 )
                    rx = ry = rz = 0;
                else
                {
                    t = (R(0, 0) + 1)*0.5;
                    rx = sqrt( std::max(t, 0.0) );
                    t = (R(1, 1) + 1)*0.5;
                    ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
                    t = (R(2, 2) + 1)*0.5;
                    rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

                    if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                        rz = -rz;
                    theta /= sqrt(rx*rx + ry*ry + rz*rz);
                    rx *= theta;
                    ry *= theta;
                    rz *= theta;
                }
            }
            else
            {
                double vth = 1/(2*s);
                vth *= theta;
                rx *= vth; ry *= vth; rz *= vth;
            }
            return Eigen::Vector3d(rx, ry, rz).cast<float>();

        }

};

#endif /* ODOMETRYPROVIDER_H_ */

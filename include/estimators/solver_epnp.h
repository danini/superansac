// Corrected EPnPSolver implementation
#pragma once

#include <Eigen/Dense>
#include <vector>
#include "abstract_solver.h"
#include "../utils/math_utils.h"
#include "../models/model.h"
#include "../utils/types.h"

namespace superansac
{
    namespace estimator
    {
        namespace solver
        {
            // This is the estimator class for the EPnP algorithm, which estimates
            // the camera pose (rotation and translation) from n 3D-2D point correspondences.
            class EPnPSolver : public AbstractSolver
            {
            public:
                EPnPSolver()
                {
                }

                ~EPnPSolver()
                {
                }

                // Determines if there is a chance of returning multiple models
                // when the function 'estimateModel' is applied.
                bool returnMultipleModels() const override
                {
                    return false;
                }

                // The maximum number of solutions returned by the estimator
                size_t maximumSolutions() const override
                {
                    return 1;
                }

                // The minimum number of points required for the estimation
                size_t sampleSize() const override
                {
                    return 4;
                }

                // Estimate the model parameters from the given point sample
                // using weighted fitting if possible.
                bool estimateModel(
                    const DataMatrix& kData_, // The set of data points
                    const size_t *kSample_,   // The sample used for the estimation
                    const size_t kSampleNumber_, // The size of the sample
                    std::vector<models::Model> &models_, // The estimated model parameters
                    const double *kWeights_ = nullptr) const override; // The weight for each point

            protected:
                // Chooses control points for the EPnP algorithm
                void chooseControlPoints(
                    const Eigen::MatrixXd& worldPoints,
                    Eigen::Matrix<double, 4, 3>& controlPoints) const;

                // Computes the barycentric coordinates of the world points
                bool computeBarycentricCoordinates(
                    const Eigen::MatrixXd& worldPoints,
                    const Eigen::Matrix<double, 4, 3>& controlPoints,
                    Eigen::MatrixXd& alphas) const;

                // Constructs the M matrix used in EPnP
                void fillM(
                    const Eigen::MatrixXd& alphas,
                    const Eigen::MatrixXd& imagePoints,
                    const double focalLengthU,
                    const double focalLengthV,
                    const double principalPointU,
                    const double principalPointV,
                    Eigen::MatrixXd& M) const;

                // Solves for the camera pose using EPnP
                bool solveForPose(
                    const Eigen::Matrix<double, 12, 12>& MtM,
                    const Eigen::Matrix<double, 4, 3>& controlPoints,
                    models::Model& model) const;

                // Computes the L matrix and rho vector for solving betas
                void computeL6x10(
                    const Eigen::Matrix<double, 12, 12>& Ut,
                    Eigen::Matrix<double, 6, 10>& L6x10) const;

                void computeRho(
                    const Eigen::Matrix<double, 4, 3>& controlPoints,
                    Eigen::Matrix<double, 6, 1>& rho) const;

                // Solves for betas using the null space of M
                void findBetas(
                    const Eigen::Matrix<double, 6, 10>& L6x10,
                    const Eigen::Matrix<double, 6, 1>& rho,
                    Eigen::Vector4d& betas) const;

                // Estimates rotation and translation from control points
                void estimateRandT(
                    const Eigen::Matrix<double, 4, 3>& cameraControlPoints,
                    const Eigen::Matrix<double, 4, 3>& controlPoints,
                    Eigen::Matrix3d& R,
                    Eigen::Vector3d& t) const;
            };

            bool EPnPSolver::estimateModel(
                const DataMatrix& kData_, // The set of data points
                const size_t *kSample_,   // The sample used for the estimation
                const size_t kSampleNumber_, // The size of the sample
                std::vector<models::Model> &models_, // The estimated model parameters
                const double *kWeights_) const
            {
                // Ensure that we have at least 4 correspondences
                if (kSampleNumber_ < sampleSize())
                    return false;

                // Reuse thread-local buffers across LO fits (eliminates 5 heap allocs/frees per call)
                static thread_local Eigen::MatrixXd worldPoints;
                static thread_local Eigen::MatrixXd imagePoints;
                static thread_local Eigen::MatrixXd alphas;
                worldPoints.resize(kSampleNumber_, 3);
                imagePoints.resize(kSampleNumber_, 2);

                const size_t kCols = static_cast<size_t>(kData_.cols());
                const double * __restrict__ kBase = kData_.data();
                for (size_t i = 0; i < kSampleNumber_; ++i)
                {
                    const size_t idx = kSample_ == nullptr ? i : kSample_[i];
                    const double * __restrict__ row = kBase + idx * kCols;
                    imagePoints(i, 0) = row[0];
                    imagePoints(i, 1) = row[1];
                    worldPoints(i, 0) = row[2];
                    worldPoints(i, 1) = row[3];
                    worldPoints(i, 2) = row[4];
                }

                // Choose control points
                Eigen::Matrix<double, 4, 3> controlPoints;
                chooseControlPoints(worldPoints, controlPoints);
                if (!controlPoints.allFinite())
                    return false;

                // Compute barycentric coordinates
                alphas.resize(kSampleNumber_, 4);
                if (!computeBarycentricCoordinates(worldPoints, controlPoints, alphas))
                    return false;

                // Accumulate the 12x12 symmetric normal matrix M^T * M directly from its 4x4 block-sparse
                // Kronecker structure (each 3x3 block (p, q) is sum_i alpha_{ip}*alpha_{iq} * [1,0,-u; 0,1,-v; -u,-v,u^2+v^2]),
                // avoiding materializing the 2N x 12 matrix M and reducing work from 288*N to 40*N muls.
                double s1[10] = {}, su[10] = {}, sv[10] = {}, sr2[10] = {};
                for (size_t i = 0; i < kSampleNumber_; ++i)
                {
                    const double a0 = alphas(i, 0), a1 = alphas(i, 1), a2 = alphas(i, 2), a3 = alphas(i, 3);
                    const double u = imagePoints(i, 0), v = imagePoints(i, 1);
                    const double r2 = u * u + v * v;
                    const double a[4] = {a0, a1, a2, a3};
                    int k = 0;
                    for (int p = 0; p < 4; ++p)
                    {
                        const double ap = a[p];
                        for (int q = 0; q <= p; ++q, ++k)
                        {
                            const double w = ap * a[q];
                            s1[k]  += w;
                            su[k]  -= w * u;
                            sv[k]  -= w * v;
                            sr2[k] += w * r2;
                        }
                    }
                }

                Eigen::Matrix<double, 12, 12> MtM = Eigen::Matrix<double, 12, 12>::Zero();
                int k = 0;
                for (int p = 0; p < 4; ++p)
                {
                    for (int q = 0; q <= p; ++q, ++k)
                    {
                        const int rp = 3 * p, cq = 3 * q;
                        MtM(rp,     cq)     = s1[k];
                        MtM(rp + 1, cq + 1) = s1[k];
                        MtM(rp,     cq + 2) = su[k];
                        MtM(rp + 2, cq)     = su[k];
                        MtM(rp + 1, cq + 2) = sv[k];
                        MtM(rp + 2, cq + 1) = sv[k];
                        MtM(rp + 2, cq + 2) = sr2[k];
                        if (p != q)
                        {
                            MtM.block<3, 3>(cq, rp) = MtM.block<3, 3>(rp, cq);
                        }
                    }
                }
                if (!MtM.allFinite())
                    return false;

                // Solve for pose
                models::Model model;
                if (!solveForPose(MtM, controlPoints, model))
                    return false;
                if (!model.getData().allFinite())
                    return false;

                models_.emplace_back(model);
                return true;
            }

            void EPnPSolver::chooseControlPoints(
                const Eigen::MatrixXd& worldPoints,
                Eigen::Matrix<double, 4, 3>& controlPoints) const
            {
                // Compute centroid of world points
                Eigen::Vector3d centroid = worldPoints.colwise().mean();

                controlPoints.row(0) = centroid.transpose();

                // Compute principal components (PCA)
                static thread_local Eigen::MatrixXd PW0;
                PW0.resize(worldPoints.rows(), 3);
                PW0 = worldPoints.rowwise() - centroid.transpose();

                Eigen::JacobiSVD<Eigen::MatrixXd> svd(PW0, Eigen::ComputeThinV);
                Eigen::Matrix3d V = svd.matrixV();
                Eigen::Vector3d S = svd.singularValues();

                // Scale principal components based on singular values
                const double invRows = 1.0 / static_cast<double>(worldPoints.rows());
                for (int i = 0; i < 3; ++i)
                    controlPoints.row(i + 1) = centroid.transpose() + V.col(i).transpose() * (S(i) * invRows);
            }

            bool EPnPSolver::computeBarycentricCoordinates(
                const Eigen::MatrixXd& worldPoints,
                const Eigen::Matrix<double, 4, 3>& controlPoints,
                Eigen::MatrixXd& alphas) const
            {
                Eigen::Matrix3d CC;
                for (int i = 0; i < 3; ++i)
                    CC.col(i) = controlPoints.row(i + 1).transpose() - controlPoints.row(0).transpose();

                if (!CC.allFinite() || std::abs(CC.determinant()) < 1e-9)
                    return false;

                Eigen::Matrix3d CC_inv = CC.inverse();
                if (!CC_inv.allFinite())
                    return false;
                for (int i = 0; i < worldPoints.rows(); ++i)
                {
                    Eigen::Vector3d pi = worldPoints.row(i).transpose() - controlPoints.row(0).transpose();
                    Eigen::Vector3d alpha = CC_inv * pi;
                    alphas(i, 1) = alpha(0);
                    alphas(i, 2) = alpha(1);
                    alphas(i, 3) = alpha(2);
                    alphas(i, 0) = 1.0 - alpha.sum();
                }
                return alphas.allFinite();
            }

            void EPnPSolver::fillM(
                const Eigen::MatrixXd& alphas,
                const Eigen::MatrixXd& imagePoints,
                const double fu,
                const double fv,
                const double uc,
                const double vc,
                Eigen::MatrixXd& M) const
            {
                for (int i = 0; i < alphas.rows(); ++i)
                {
                    const Eigen::Vector4d alpha = alphas.row(i).transpose();
                    const double u = imagePoints(i, 0);
                    const double v = imagePoints(i, 1);

                    for (int j = 0; j < 4; ++j)
                    {
                        M(2 * i, 3 * j)     = alpha(j) * fu;
                        M(2 * i, 3 * j + 1) = 0.0;
                        M(2 * i, 3 * j + 2) = alpha(j) * (uc - u);

                        M(2 * i + 1, 3 * j)     = 0.0;
                        M(2 * i + 1, 3 * j + 1) = alpha(j) * fv;
                        M(2 * i + 1, 3 * j + 2) = alpha(j) * (vc - v);
                    }
                }
            }

            void EPnPSolver::computeL6x10(
                const Eigen::Matrix<double, 12, 12>& Ut,
                Eigen::Matrix<double, 6, 10>& L6x10) const
            {
                // The 4 vectors from the last 4 columns of Ut
                const Eigen::Matrix<double, 12, 4> V = Ut.rightCols<4>();

                Eigen::Vector3d dv[6][4];
                int idx = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = i + 1; j < 4; ++j)
                    {
                        for (int k = 0; k < 4; ++k)
                        {
                            dv[idx][k] = V.block<3, 1>(3 * i, k) - V.block<3, 1>(3 * j, k);
                        }
                        idx++;
                    }
                }

                for (int i = 0; i < 6; ++i)
                {
                    int row = i;
                    int col = 0;
                    for (int j = 0; j < 4; ++j)
                    {
                        for (int k = j; k < 4; ++k)
                        {
                            double value = dv[i][j].dot(dv[i][k]);
                            if (j != k)
                                value *= 2.0;
                            L6x10(row, col++) = value;
                        }
                    }
                }
            }

            void EPnPSolver::computeRho(
                const Eigen::Matrix<double, 4, 3>& controlPoints,
                Eigen::Matrix<double, 6, 1>& rho) const
            {
                int idx = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = i + 1; j < 4; ++j)
                    {
                        rho(idx++) = (controlPoints.row(i) - controlPoints.row(j)).squaredNorm();
                    }
                }
            }

            void EPnPSolver::findBetas(
                const Eigen::Matrix<double, 6, 10>& L6x10,
                const Eigen::Matrix<double, 6, 1>& rho,
                Eigen::Vector4d& betas) const
            {
                // Solve L * betas = rho using fixed-size SVD (zero heap allocations)
                Eigen::JacobiSVD<Eigen::Matrix<double, 6, 10>> svd(L6x10, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix<double, 10, 1> betas_10 = svd.solve(rho);

                // Extract betas from betas_10
                betas(0) = sqrt(fabs(betas_10(0)));
                betas(1) = sqrt(fabs(betas_10(2)));
                betas(2) = sqrt(fabs(betas_10(5)));
                betas(3) = sqrt(fabs(betas_10(9)));

                // Adjust signs if necessary
                if (betas_10(1) < 0) betas(1) = -betas(1);
                if (betas_10(3) < 0) betas(2) = -betas(2);
                if (betas_10(6) < 0) betas(3) = -betas(3);
            }

            bool EPnPSolver::solveForPose(
                const Eigen::Matrix<double, 12, 12>& MtM,
                const Eigen::Matrix<double, 4, 3>& controlPoints,
                models::Model& model) const
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(MtM);
                if (es.info() != Eigen::Success || !es.eigenvectors().allFinite())
                    return false;
                // Eigenvectors of SelfAdjointEigenSolver are sorted in increasing eigenvalue order,
                // while computeL6x10 expects the 4 smallest-singular-value vectors in the last 4 columns.
                Eigen::Matrix<double, 12, 12> V;
                for (int c = 0; c < 12; ++c)
                    V.col(c) = es.eigenvectors().col(11 - c);

                // Compute L6x10 and rho
                Eigen::Matrix<double, 6, 10> L6x10;
                computeL6x10(V, L6x10);

                Eigen::Matrix<double, 6, 1> rho;
                computeRho(controlPoints, rho);
                if (!L6x10.allFinite() || !rho.allFinite())
                    return false;

                // Solve for betas
                Eigen::Vector4d betas;
                findBetas(L6x10, rho, betas);
                if (!betas.allFinite())
                    return false;

                // Compute camera control points
                Eigen::Matrix<double, 4, 3> cameraControlPoints = Eigen::Matrix<double, 4, 3>::Zero();
                for (int i = 0; i < 4; ++i)
                {
                    const Eigen::Matrix<double, 12, 1> v = V.col(11 - i);
                    for (int j = 0; j < 4; ++j)
                    {
                        cameraControlPoints.row(j) += betas(i) * v.segment<3>(3 * j).transpose();
                    }
                }
                if (cameraControlPoints.col(2).mean() < 0.0)
                    cameraControlPoints = -cameraControlPoints;

                // Estimate rotation and translation
                Eigen::Matrix3d R;
                Eigen::Vector3d t;
                estimateRandT(cameraControlPoints, controlPoints, R, t);

                // Build model
                auto& modelData = model.getMutableData();
                modelData.resize(3, 4);
                modelData.block<3, 3>(0, 0) = R;
                modelData.col(3) = t;
                return modelData.allFinite();
            }

            void EPnPSolver::estimateRandT(
                const Eigen::Matrix<double, 4, 3>& cameraControlPoints,
                const Eigen::Matrix<double, 4, 3>& controlPoints,
                Eigen::Matrix3d& R,
                Eigen::Vector3d& t) const
            {
                // Compute centroids
                Eigen::Vector3d cc_centroid = cameraControlPoints.colwise().mean();
                Eigen::Vector3d wc_centroid = controlPoints.colwise().mean();

                // Compute cross-covariance matrix
                Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
                for (int i = 0; i < 4; ++i)
                {
                    Eigen::Vector3d cc = cameraControlPoints.row(i).transpose() - cc_centroid;
                    Eigen::Vector3d wc = controlPoints.row(i).transpose() - wc_centroid;
                    H += cc * wc.transpose();
                }

                // Compute rotation using SVD
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
                R = svd.matrixU() * svd.matrixV().transpose();

                // Ensure a proper rotation (determinant = 1)
                if (R.determinant() < 0)
                {
                    Eigen::Matrix3d U = svd.matrixU();
                    U.col(2) *= -1;
                    R = U * svd.matrixV().transpose();
                }

                // Compute translation
                t = cc_centroid - R * wc_centroid;
            }

        } // namespace solver
    }     // namespace estimator
} // namespace superansac

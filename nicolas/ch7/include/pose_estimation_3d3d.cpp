/* ======== Measurement model (Observation Equation) ======== */
/*  z_k: Image Measurements (p)                               */
/*  x_k: Camera Pose (T)                                      */
/*  y_kj: 3D Points (P)                                       */
/*                                                            */
/*    z_kj = h(y_kj, x_k) + v_kj             (Classic SLAM)   */
/*  s*z_kj = K*T_k*P_kj = K*(R_k*P_kj+t_k)          (vSLAM)   */
/*                                                            */
/*  Simplification:                                           */
/*    z_j = h(T, P_j)                               (vSLAM)   */
/* ========================================================== */

/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <chrono>

/* Eigen3 Libraries */
#include <eigen3/Eigen/Core>

/* OpenCV Libraries */
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

/* g2o Libraries */
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

/* Sophus Libraries */
#include <sophus/se3.hpp>

/* Custom Libraries */
#include "../../common/libUtils_basic.h"
#include "../../common/libUtils_eigen.h"
#include "../../common/libUtils_opencv.h"

using namespace std;
using namespace cv;

/* Global Variables */
// The memory is aligned as for dynamically aligned matrix/array types such as MatrixXd
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Choose the optimization algorithm:
const char* optimization_methods_enum2str[] = {"Gauss-Newton", "Levenberg-Marquardt", "Powell's Dog Leg"};
int optimization_method_selected = 1;

/* =========== */
/*  Functions  */
/* =========== */
void ICP_SVD(const vector<Point3f> &pts1_p, const vector<Point3f> &pts2_p, Mat &R, Mat &t){
    cout << "| ----------- |" << endl;
    cout << "|  ICP (SVD)  |" << endl;
    cout << "| ----------- |" << endl;
    // /* 1. Calculate the centroids of the two groups of points p, p', and then calculate the de-centroid coordinates of each point: */
    // // Centroids:
    // //   p = 1/n*sum_i(p_i), p' = 1/n*sum_i(p'_i)
    // Point3f p1_cent, p2_cent;  // Centroids, p and p'

    // int N = pts1_p.size();
    // for(int i=0; i < N; i++){
    //     p1_cent += pts1_p[i];
    //     p2_cent += pts2_p[i];
    // }

    // // p1_cent = p1_cent/float(N);
    // // p2_cent = p2_cent/float(N);

    // // FIXME: Why make these conversions?
    // p1_cent = Point3f(Vec3f(p1_cent) / N);
    // p2_cent = Point3f(Vec3f(p2_cent) / N);

    // cout << "Centroids:" << endl;
    // cout << "-- p1_cent: " << p1_cent << endl;
    // cout << "-- p2_cent: " << p2_cent << endl << endl;

    // // De-centroid coordinates: 
    // //   q_i = p_i − p, q'_i = p'_i − p'.
    // vector<Point3f> pts1_q(N), pts2_q(N);  // {q_i}_n, {q'_i}_n
    
    // for(int i=0; i < N; i++){
    //     pts1_q[i] = pts1_p[i] - p1_cent;  // q_i
    //     pts2_q[i] = pts2_p[i] - p2_cent;  // q'_i
    // }

    // /* 2. The rotation matrix is calculated according to the following optimization problem: */
    // // R∗ = argmin_R 0.5*sum_i(||q_i-R*q'_i||^2)
    // // Next, we introduce how to solve the optimal R in the above minimization problem through SVD;
    // // Let's Compute W = sum_i(q_i*q'_i^T), W is a 3-by-3 matrix.
    // Eigen::Matrix3d W = Eigen::Matrix3d::Zero();

    // for(int i=0; i < N; i++){
    //     // Since q_i, q'_i are Point3f, it's necessary to convert to the Eigen's Vector3d.
    //     W += Eigen::Vector3d(pts1_q[i].x, pts1_q[i].y, pts1_q[i].z)*Eigen::Vector3d(pts2_q[i].x, pts2_q[i].y, pts2_q[i].z).transpose();
    // }
    
    // printMatrix<Eigen::Matrix3d>("W: ", W);

    // // Decompose W, W = UΣV^T
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::Matrix3d U = svd.matrixU();
    // Eigen::Matrix3d V = svd.matrixV();

    // printMatrix<Eigen::Matrix3d>("U: ", U);
    // printMatrix<Eigen::Matrix3d>("V: ", V);

    // // Calculate R, R = U*V^T
    // Eigen::Matrix3d R_ = U*(V.transpose());

    // // Check if the det(R_) is negative
    // if(R_.determinant() < 0) 
    //     R_ = -R_;

    // /* 3. Calculate t according to R in step 2: */
    // // t∗ = p − Rp'.
    // Eigen::Vector3d t_ = Eigen::Vector3d(p1_cent.x, p1_cent.y, p1_cent.z) - R_*Eigen::Vector3d(p2_cent.x, p2_cent.y, p2_cent.z);

    // /* Convert Eigen::Vector3d back to cv::Mat */
    // R = (Mat_<double>(3, 3) << 
    //     R_(0, 0), R_(0, 1), R_(0, 2), 
    //     R_(1, 0), R_(1, 1), R_(1, 2), 
    //     R_(2, 0), R_(2, 1), R_(2, 2)
    // );
    // t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));



    Point3f p1_cent, p2_cent;     // center of mass
    int N = pts1_p.size();
    for (int i = 0; i < N; i++) {
        p1_cent += pts1_p[i];
        p2_cent += pts2_p[i];
    }
    p1_cent = Point3f(Vec3f(p1_cent) / N);
    p2_cent = Point3f(Vec3f(p2_cent) / N);

    cout << "Centroids: " << endl;
    cout << p1_cent << endl;
    cout << p2_cent << endl;

    vector<Point3f> pts1_q(N), pts2_q(N); // remove the center
    for (int i = 0; i < N; i++) {
        pts1_q[i] = pts1_p[i] - p1_cent;
        pts2_q[i] = pts2_p[i] - p2_cent;
    }


     // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(pts1_q[i].x, pts1_q[i].y, pts1_q[i].z) * Eigen::Vector3d(pts2_q[i].x, pts2_q[i].y, pts2_q[i].z).transpose();
    }
    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1_cent.x, p1_cent.y, p1_cent.z) - R_ * Eigen::Vector3d(p2_cent.x, p2_cent.y, p2_cent.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

    

    

    /* Results */
    cout << "ICP via SVD results: " << endl;
    printMatrix("R:\n", R);
    printMatrix("t:\n", t);
    printMatrix("R_inv:\n", R.t());
    printMatrix("t_inv:\n", -R.t()*t);
}

/* ------------ */
/*  PoseVertex  */
/* ------------ */
/* Description: The vertex of the 6D Camera Pose
/* Template parameters: optimized variable dimensions and data types
*/
class PoseVertex : public g2o::BaseVertex<6, Sophus::SE3d>{  // Inheritance of the class "g2o::BaseVertex"
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // Reset
    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    // Update, Left Multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        // Conversion to Eigen data type
        Vector6d dx = Vector6d(update);  // δξ

        // Left multiply T by a disturbance quantity exp(δξ)
        _estimate = Sophus::SE3d::exp(dx)*_estimate;  // T_k = exp(δξ_k).T_{k-1}
    }

    // Save and read: leave blank
    virtual bool read(istream &in) {return 0;}

    virtual bool write(ostream &out) const {return 0;}
};

/* ---------------------------- */
/*  ProjectXYZRGBDPoseOnlyEdge  */
/* ---------------------------- */
/* Description: Minimizes the 3D-3D Points Matching Error model, e = p_i - R*p'_i
/* Template parameters: measurement dimension, measurement type, connection vertex type
*/
class ProjectXYZRGBDPoseOnlyEdge : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, PoseVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /** Constructor
     * 1. This is a construct declaration when using the operator () to pass a given parameter, in this case a double variable "x". Construct a new Projection Edge object
     * 2. Since this class inherited the "BaseUnaryEdge" class, we also need to initialized it.
     * 3. The "_x(x)" stores the passed value passed to "x" on the class private attribute "_x".
     */
    ProjectXYZRGBDPoseOnlyEdge(const Eigen::Vector3d &P2) : _P2(P2) {}

    // Calculate the 3D-3D Matching error
    virtual void computeError() override {
        // Creates a pointer to the already created graph's vertice (Node, id=0), which allows to access and retrieve the values of "T".
        const PoseVertex *v = static_cast<PoseVertex *> (_vertices[0]);  // _vertices was inherited from g2o::BaseUnaryEdge
        Sophus::SE3d T = v->estimate();                                  // Get estimated value, T*

        // Calculate the residual
        // Residual = Y_real - Y_estimated = _measurement - h(_x, _estimate)
        _error = _measurement - T*_P2;  // e = p_i - R*p'_i
    }

    /** Calculate the Jacobian matrix
     * Linearizes the oplus operator in the vertex, and stores  the result in temporary variables _jacobianOplusXi and _jacobianOplusXj
     */
    virtual void linearizeOplus() override{
        // Creates a pointer to the already created graph's vertice (Node, id=0), which allows to access and retrieve the values of "T".
        const PoseVertex *v = static_cast<PoseVertex *> (_vertices[0]);  // _vertices was inherited from g2o::BaseUnaryEdge
        Sophus::SE3d T = v->estimate();                                  // Get estimated value, T*

        // Describe the 3D Space Point P2 in the {camera1} frame using the T*
        Eigen::Vector3d P1_est = T*_P2;  // P2 = T*.P1 = [X, Y, Z]^T

        /* ----- Compute Jacobian Matrix ----- */
        // The jacobian matrix indicates how the error varies according to the increment δξ, ∂e/∂δξ
        _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();  // TODO: Why?
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(P1_est);  // TODO: Why?
    }

    virtual bool read(istream &in) override {return 0;}

    virtual bool write(ostream &out) const override {return 0;}

private:
    Eigen::Vector3d _P2;
};

void ICP_bundleAdjustment(const vector<Point3f> &pts1_p, const vector<Point3f> &pts2_p, Mat &R, Mat &t){
    cout << "| --------------- |" << endl;
    cout << "|  ICP (BA, g2o)  |" << endl;
    cout << "| --------------- |" << endl;

    /* ----- Build Graph for Optimization ----- */
    // First, let's set g2o
    // typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;            // Pose is 6D, Landmarks are 3D (3D Points)
    typedef g2o::BlockSolverX BlockSolverType;  //! variable size solver FIXME: Why?
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  // Solver type: Linear
    
    
    // Gradient descent method, you can choose from GN (Gauss-Newton), LM(Levenberg-Marquardt), Powell's dog leg methods.
    g2o::OptimizationAlgorithmWithHessian *solver;

    cout << "Graph Optimization Algorithm selected: " << optimization_methods_enum2str[optimization_method_selected-1] << endl << endl;

    switch (optimization_method_selected){
        case 1:  // Option 1: Gauss-Newton method
            solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            break;
        case 2:  // Option 2: Levenberg-Marquardt method
            solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            break;
        case 3:  // Option 3: Powell's Dog Leg Method
            solver = new g2o::OptimizationAlgorithmDogleg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            break;
        default:
            break;
    }

    // Configure the optimizer
    g2o::SparseOptimizer optimizer;  // Graph Model
    optimizer.setAlgorithm(solver);  // Set up the solver
    optimizer.setVerbose(true);      // Turn on debugging output

    // Add vertices to the graph
    PoseVertex *poseVertex = new PoseVertex();  // Camera Pose Vertex
    poseVertex->setId(0);                       //! Sets the id of the node in the graph, be sure that the graph keeps consistent after changing the id
    poseVertex->setEstimate(Sophus::SE3d());    //! Sets the estimate for the vertex also calls g2o::OptimizableGraph::Vertex::updateCache()
    optimizer.addVertex(poseVertex);

    // Add edges to the graph
    // int index = 1;
    for(size_t i=0; i<pts1_p.size(); ++i){
        // cv::Point3f to Eigen::Vector3d
        Vector3d P1_3d(pts1_p[i].x, pts1_p[i].y, pts2_p[i].z);  // P1_i, p_i
        Vector3d P2_3d(pts2_p[i].x, pts2_p[i].y, pts2_p[i].z);  // P2_i, p'_i
        
        ProjectXYZRGBDPoseOnlyEdge *projEdge = new ProjectXYZRGBDPoseOnlyEdge(P2_3d);  // Creates the i-th edge
        // projEdge->setId(index);                                   // Specifies the edge ID  // FIXME: the original code doesn't have this line. Why?
        projEdge->setVertex(0, poseVertex);                       // Connects edge to the vertex (Node, 0)
        projEdge->setMeasurement(P1_3d);                          // Observed value
        projEdge->setInformation(Eigen::Matrix3d::Identity());    // Information matrix: the inverse of the covariance matrix
        optimizer.addEdge(projEdge);
        // index++;
    }

    /* ----- Solve! ----- */
    cout << "Summary: " << endl;
    Timer t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();      // Start!
    optimizer.optimize(10);                  // Number of optimization steps
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);

    /* ----- Results ----- */
    Sophus::SE3d pose = poseVertex->estimate();  // T*
    Matrix3d R_ = pose.rotationMatrix();
    Vector3d t_ = pose.translation();

     /* Convert Eigen::Vector3d back to cv::Mat */
    R = (Mat_<double>(3, 3) << 
        R_(0, 0), R_(0, 1), R_(0, 2), 
        R_(1, 0), R_(1, 1), R_(1, 2), 
        R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
    
    printMatrix<Matrix4d>("\nT* (g2o):\n", pose.matrix());
    printMatrix("R:\n", R);
    printMatrix("t:\n", t);
}

// TODO: Excluir codigos abaixo ----------------------------------------------------------------------------------------------------------------------

    // cout << endl << "after optimization:" << endl;
    // cout << "T=\n" << pose->estimate().matrix() << endl;

    // // convert to cv::Mat
    // Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    // Eigen::Vector3d t_ = pose->estimate().translation();
    // R = (Mat_<double>(3, 3) <<
    //     R_(0, 0), R_(0, 1), R_(0, 2),
    //     R_(1, 0), R_(1, 1), R_(1, 2),
    //     R_(2, 0), R_(2, 1), R_(2, 2)
    // );
    // t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));

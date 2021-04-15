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
Eigen::Matrix<double, 2, 6> calculateJacobian(const Eigen::Vector3d &P2, const Eigen::Matrix3d &K){
    double fx = K(0,0);
    double fy = K(1,1);
    // double cx = K(0,2);
    // double cy = K(1,2);

    double X = P2[0];        
    double Y = P2[1];
    double Z = P2[2];
        
    double X2 = X*X;
    double Y2 = Y*Y;
    double Z2 = Z*Z;
        
    Eigen::Matrix<double, 2, 6> J;
    J << -fx/Z,     0, fx*X/Z2,   fx*X*Y/Z2, -fx-fx*X2/Z2,  fx*Y/Z,
             0, -fy/Z, fy*Y/Z2, fy+fy*Y2/Z2,   -fy*X*Y/Z2, -fy*X/Z;

    return J;
}

void bundleAdjustmentGaussNewton(const VecVector3d &pts1_3d, const VecVector2d &pts2_2d, const Eigen::Matrix3d &K, Sophus::SE3d &pose){
    cout << "| ----------------------- |" << endl;
    cout << "|  Bundle Adjustment (GN) |" << endl;
    cout << "| ----------------------- |" << endl;

    /* Initialization */
    const int iterations = 10;
    double lastCost = 0.0;
    
    double fx = K(0, 0);
    double fy = K(1, 1);
    double cx = K(0, 2);
    double cy = K(1, 2);

    /* Iteration Loop */
    cout << "Summary: " << endl;
    Timer t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++){
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        double cost = 0.0;  // Reset

        /* Data Loop, Compute Cost */
        for (int i = 0; i < pts1_3d.size(); i++){
            /* ----- Coordinate System Transformation ----- */
            // Describe the 3D Space Point P @ {camera1} in the {camera2} frame
            Eigen::Vector3d P2 = pose * pts1_3d[i]; // P'_i = (T*.P_i)1:3, P2_i = T21*.P1_i

            /* ----- Compute Reprojection Error ----- */
            // Compute the Estimated Projection of P2 in Camera 2's Image Plane (Pixel Coordinates)
            Eigen::Vector2d proj = cam2pixel(P2, K);  // p2^=[u2, v2]^T = [fx*X/Z+cx, fy*Y/Z+cy]^T

            // Compute Residual
            Eigen::Vector2d e = pts2_2d[i] - proj; // e = p2_i - p2^_i
            
            // Compute the Least-Squares Cost 
            // This is the actual error function being minimized (Objective Function) by 
            // solving the proposed linear system: min_x(sum_i ||ei(x)||^2).
            cost += e.squaredNorm();  // Summation of the squared residuals
        
            /* ----- Compute Jacobian Matrix ----- */
            // The jacobian matrix indicates how the error varies according to the increment δξ, ∂e/∂δξ
            Eigen::Matrix<double, 2, 6> J = calculateJacobian(P2, K);

            /* ----- Hessian and Bias ----- */
            // Information Matrix(Ω) wasn't informed, so consider it as identity.
            H +=  J.transpose() * J;  // Hessian, H(x) = J(x)'*Ω*J(x)
            b += -J.transpose() * e;  // Bias, g(x) = -b(x) = -Ω*J(x)*f(x), f(x)=e(x)
        }

        /* ----- Solve! ----- */
        // Solve the Linear System A*x=b, H(x)*∆x = g(x)
        Vector6d dx = H.ldlt().solve(b);  // δξ (Lie Algebra)

        // Check Solution
        if (isnan(dx[0])){
            cout << "Result is nan!" << endl;
            break;
        }

        /* Stopping Criteria */
        // If the cost increased, the update was not good, then break.
        if (iter > 0 && cost >= lastCost){
            cout << "\ncost: " << cost << " >= lastCost: " << lastCost << ", break!" << endl;
            break;
        }

        /* ----- Update ----- */
        // Left multiply T by a disturbance quantity exp(δξ)
        pose = Sophus::SE3d::exp(dx) * pose;  // T* = exp(δξ).T
        lastCost = cost;
        
        cout << "it: " << iter << ",\tcost: " << std::setprecision(12) << cost << ",\tupdate: " << dx.transpose() << endl;
        
        if (dx.norm() < 1e-6){  // Method converged!
            break;
        }
    }
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);

    /* ----- Results ----- */
    cout << "\nPose (T*) by GN:\n" << pose.matrix() << endl << endl;
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

/* ---------------- */
/*  ProjectionEdge  */
/* ---------------- */
/* Description: Minimizes the 3D-2D Points Reprojection Error model, e = p2 - p2^ = [e_u, e_v]
/* Template parameters: measurement dimension, measurement type, connection vertex type
*/
class ProjectionEdge : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, PoseVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /** Constructor
     * 1. This is a construct declaration when using the operator () to pass a given parameter, in this case a double variable "x". Construct a new Projection Edge object
     * 2. Since this class inherited the "BaseUnaryEdge" class, we also need to initialized it.
     * 3. The "_x(x)" stores the passed value passed to "x" on the class private attribute "_x".
     */
    ProjectionEdge(const Eigen::Vector3d &P1, const Eigen::Matrix3d &K) : _P1(P1), _K(K) {}

    // Calculate the reprojection error
    virtual void computeError() override {
        // Creates a pointer to the already created graph's vertice (Node, id=0), which allows to access and retrieve the values of "T".
        const PoseVertex *v = static_cast<const PoseVertex *> (_vertices[0]);  // _vertices was inherited from g2o::BaseUnaryEdge
        Sophus::SE3d T = v->estimate();                                        // Get estimated value, T*

        // Describe the 3D Space Point P in the {camera2} frame
        Eigen::Vector3d p2_proj = _K*(T*_P1);  // s2.p2^ = K.(T*.P1) = K.P2 = [s2.u2, s2.v2, s2]^T
        p2_proj /= p2_proj[2];                    //    p2^ = [u2, v2, 1]^T
        
        // Calculate the residual
        // Residual = Y_real - Y_estimated = _measurement - h(_x, _estimate)
        _error = _measurement - p2_proj.head<2>();  // e = p2 - p2^ = [eu, ev]^T
    }

    /** Calculate the Jacobian matrix
     * Linearizes the oplus operator in the vertex, and stores  the result in temporary variables _jacobianOplusXi and _jacobianOplusXj
     */
    virtual void linearizeOplus() override{
        // Creates a pointer to the already created graph's vertice (Node, id=0), which allows to access and retrieve the values of "T".
        const PoseVertex *v = static_cast<const PoseVertex *> (_vertices[0]);  // _vertices was inherited from g2o::BaseUnaryEdge
        Sophus::SE3d T = v->estimate();                                        // Get estimated value, T*
        
        // Describe the 3D Space Point P in the {camera2} frame
        Eigen::Vector3d P2 = T*_P1;  // P2 = T*.P1 = [X, Y, Z]^T
        
        /* ----- Compute Jacobian Matrix ----- */
        // The jacobian matrix indicates how the error varies according to the increment δξ, ∂e/∂δξ
        _jacobianOplusXi << calculateJacobian(P2, _K);
    }

    virtual bool read(istream &in) override {return 0;}

    virtual bool write(ostream &out) const override {return 0;}

private:
    Eigen::Vector3d _P1;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(const VecVector3d &pts1_3d, const VecVector2d &pts2_2d, const Eigen::Matrix3d &K, Sophus::SE3d &pose){
    cout << "| ------------------------- |" << endl;
    cout << "|  Bundle Adjustment (g2o)  |" << endl;
    cout << "| ------------------------- |" << endl;

    /* ----- Build Graph for Optimization ----- */
    // First, let's set g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;            // Pose is 6D, Landmark are 3D (3D Points)
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
    int index = 1;
    for(size_t i=0; i<pts2_2d.size(); ++i){
        auto P1_3d = pts1_3d[i];  // P1_i
        auto p2_2d = pts2_2d[i];  // p2_i

        ProjectionEdge *projEdge = new ProjectionEdge(P1_3d, K);  // Creates the i-th edge
        projEdge->setId(index);                                   // Specifies the edge ID
        projEdge->setVertex(0, poseVertex);                       // Connects edge to the vertex (Node, 0)
        projEdge->setMeasurement(p2_2d);                          // Observed value
        projEdge->setInformation(Eigen::Matrix2d::Identity());    // Information matrix: the inverse of the covariance matrix
        optimizer.addEdge(projEdge);
        index++;
    }
    
    /* ----- Solve! ----- */
    cout << "Summary: " << endl;
    Timer t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();      // Start!
    optimizer.optimize(10);                  // Number of optimization steps
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);
    
    /* ----- Results ----- */
    pose = poseVertex->estimate();
    cout << "\nPose (T*) by g2o:\n" << pose.matrix() << endl << endl;
}

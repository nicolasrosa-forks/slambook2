/* Libraries */
#include "../include/g2oCurveFitting.h"

using namespace std;

/* --------------------- */
/*  C++ Virtual Methods  */
/* --------------------- */
/* Quick Remind: A virtual function is a member function which is declared within a base class and is re-defined (Overriden) by a derived class.
/* Link: https://www.geeksforgeeks.org/virtual-function-cpp/
/* 
/* virtual & override:
/* When you override a function you don't technically need to write either virtual or override.
/* 
/* The original base class declaration needs the keyword virtual to mark it as virtual.
/* 
/* In the derived class the function is virtual by way of having the ¹same type as the base class function.
/* 
/* However, an override can help avoid bugs by producing a compilation error when the intended override isn't technically an override. For instance, the function type isn't exactly like the base class function. Or that a maintenance of the base class changes that function's type, e.g. adding a defaulted argument.
/* 
/* In the same way, a virtual keyword in the derived class can make such a bug more subtle by ensuring that the function is still virtual in the further derived classes.
/* 
/* So the general advice is,
/* 
/* Use virtual for the base class function declaration.
/* This is technically necessary.
/* 
/* Use override (only) for a derived class' override.
/* This helps maintenance.
/* 
/* Example:
/*  struct Base { virtual void foo() {} };
/*  struct Derived: Base { void foo() override {} };
*/

/* --------------------------------------- */
/*  Eigen: Structures Having Eigen Members */
/* --------------------------------------- */
/* If you define a structure having members of fixed-size vectorizable Eigen types, you must overload
/* its "operator new" so that it generates 16-bytes-aligned pointers. 
/* Fortunately, Eigen provides you with a macro EIGEN_MAKE_ALIGNED_OPERATOR_NEW that does that for you.
*/

/* -------------------- */
/*  CurveFittingVertex  */
/* -------------------- */
/* Description: The vertex of the curve model
/* Template parameters: optimized variable dimensions and data types
*/
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {  // Inheritance of the class "g2o::BaseVertex"
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  

    // Reset
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    // Update
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update);
    }

    // Save and read: leave blank
    virtual bool read(istream &in) {}
    
    virtual bool write(ostream &out) const {}
};

/* ------------------ */
/*  CurveFittingEdge  */
/* ------------------ */
/* Description: Error model 
/* Template parameters: observation dimension, type, connection vertex type
*/
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {  // Inheritance of the class "g2o::BaseUnaryEdge"
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Constructor
    // 1. This is the construct declaration when using the operator () to pass a given parameter, in this case a double variable "x".
    // 2. Since this class inherited the "BaseUnaryEdge" class we also need to initialized it.
    // 3. The "_x(x)" stores the passed value passed to "x" on the class public attribute "_x".
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // Calculate the curve model error
    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);  // Initializing conversion the a temporary variable, static_cast<new_type>(expression)
        const Eigen::Vector3d abc = v->estimate();

        // Calculation of residuals
        // Residual = Y_real - Y_estimated
        _error(0, 0) = _measurement - exp(abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0)); 
    }

    // Calculate the Jacobian matrix
    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        
        double y = exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);
        _jacobianOplusXi[0] = -_x*_x*y;
        _jacobianOplusXi[1] = -_x*y;
        _jacobianOplusXi[2] = -y;
    }

    // Save and read: leave blank
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

public:
    double _x;  // x value, y value _measurement
};

/* ====== */
/*  Main  */
/* ====== */
int main(int argc, char **argv) {
    cout << "[g2oCurveFitting] Hello!" << endl;

    /* Variables */
    double ar = 1.0, br = 2.0, cr = 1.0;        // Real parameters values
    double ae = 2.0, be = -1.0, ce = 5.0;       // Estimated parameters values
    int N = 100;                                // Number of Data points

    // cv::theRNG().state = 150;
    cv::RNG rng;                                // OpenCV Random Number generator
    double w_sigma = 1.0;                       // Noise sigma value
    double inv_sigma = 1.0 / w_sigma;

    /* ----- Data Generation ----- */
    vector<double> x_data, y_data;              // Data Vectors
  
    for(int i=0; i < N; i++){
        double x = i/100.0;
        double y = exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma);
    
        x_data.push_back(x);
        y_data.push_back(y);
    }

    printVec("x_data: ", x_data);
    cout << endl;
    printVec("y_data: ", y_data);
    cout << endl;

    /* ----- Graph Optimization ----- */
    // Build graph optimization
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;  // The optimization variable dimension of each error term is 3, and the error value dimension is 1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  // Linear Solver type

    // Gradient descent method, you can choose from GN (Gauss-Newton), LM(Levenberg-Marquardt), Powell's DogLeg
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    
    // Configure the optimizer
    g2o::SparseOptimizer optimizer;  // Graph model
    optimizer.setAlgorithm(solver);  // Set the solver
    optimizer.setVerbose(true);      // Turn on debugging output

    // Add vertices to the graph
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));  //! set the estimate for the vertex also calls g2o::OptimizableGraph::Vertex::updateCache()
    v->setId(0);                                  //! sets the id of the node in the graph be sure that the graph keeps consistent after changing the id
    optimizer.addVertex(v);

    // Add edges to the graph



    /* ----- Solve ----- */

    /* ----- Results ----- */ 


    cout << "\nDone." << endl;

    return 0;
}
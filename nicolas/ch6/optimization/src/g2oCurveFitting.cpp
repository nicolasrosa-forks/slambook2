/* Custom Libraries */
#include "../include/g2oCurveFitting.h"

using namespace std;
using namespace Eigen;

/* Global Variables */
// Choose the optimization algorithm:
// 1: Gauss-Newton, 2: Levenberg-Marquardt, 3: Powell's Dog Leg
int optimization_algorithm_selected = 1;

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
/*  struct Derived : Base { void foo() override {} };
*/

/* ----------------- */
/*  C++ static_cast  */
/* ----------------- */
/* "static_cast": static conversion
/* Ex: static_cast<new_type>(expression)
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
class CurveFittingVertex : public g2o::BaseVertex<3, Vector3d> {  // Inheritance of the class "g2o::BaseVertex"
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Reset
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    // Update
    virtual void oplusImpl(const double *update) override {
        // x_{k+1} = x_k + ∆x_k
        _estimate += Vector3d(update);  // _estimate was inherited from g2o::BaseVertex
    }

    // Save and read: leave blank
    virtual bool read(istream &in) {return 0;}

    virtual bool write(ostream &out) const {return 0;}
};

/* ------------------ */
/*  CurveFittingEdge  */
/* ------------------ */
/* Description: Curve Error model
/* Template parameters: measurement dimension, measurement type, connection vertex type
*/
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {  // Inheritance of the class "g2o::BaseUnaryEdge"
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    // 1. This is a construct declaration when using the operator () to pass a given parameter, in this case a double variable "x".
    // 2. Since this class inherited the "BaseUnaryEdge" class, we also need to initialized it.
    // 3. The "_x(x)" stores the passed value passed to "x" on the class public attribute "_x".
    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    // Calculate the curve error model
    virtual void computeError() override {
        // Creates a pointer to the already created graph's vertice (Node, id=0), which allows to access and retrieve the values of "abc".
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);  // _vertices was inherited from g2o::BaseUnaryEdge
        const Vector3d abc = v->estimate();                                                   // Get estimated value, abc*

        // Calculate the residual
        // Residual = Y_real - Y_estimated = _measurement - h(_x, _estimate)
        _error(0, 0) = _measurement - exp(abc(0, 0)*_x*_x + abc(1, 0)*_x + abc(2, 0));  // e(x) = y_r - y_e =  y_r - exp(a.x^2+b.x+c).
    }

    // Calculate the Jacobian matrix
    /**
     * Linearizes the oplus operator in the vertex, and stores the result in temporary variables _jacobianOplusXi and _jacobianOplusXj
     */
    virtual void linearizeOplus() override {
        // Creates a pointer to the already created graph's vertice (Node, id=0), which allows to access and retrieve the values of "abc".
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);  // _vertices was inherited from g2o::BaseUnaryEdge
        const Vector3d abc = v->estimate();                                                   // Get estimated value, abc*

        double y = exp(abc[0]*_x*_x + abc[1]*_x + abc[2]);  // Y_estimated

        // J(x).J(x)'.∆x = −J(x).f(x) ~= H(x).∆x = g(x)
        _jacobianOplusXi[0] = -_x*_x*y;  // de(x)/da = -x^2*exp(a*x^2 + b*x + c) = -x^2*y
        _jacobianOplusXi[1] = -_x*y;     // de(x)/db =   -x*exp(a*x^2 + b*x + c) = -x*y
        _jacobianOplusXi[2] = -y;        // de(x)/dc =   -1*exp(a*x^2 + b*x + c) = -y
    }

    // Save and read: leave blank
    virtual bool read(istream &in) {return 0;}

    virtual bool write(ostream &out) const {return 0;}

public:
    double _x;  // x value, y value (_measurement)
};

double RMSE(const Vector3d est, const Vector3d gt){
    double sum = 0.0;
    int N = 3;

    for(int i=0;i<N;i++){
        sum += pow(est[i]-gt[i], 2.0);
    }

    return sqrt(sum/(double)N);
}

/* ====== */
/*  Main  */
/* ====== */
int main(int argc, char **argv) {
    cout << "[g2oCurveFitting] Hello!" << endl;

    /* Variables */
    double ar = 1.0, br = 2.0, cr = 1.0;        // Real parameters values
    double ae = 2.0, be = -1.0, ce = 5.0;       // Estimated parameters values
    int N = 100;                                // Number of Data points

    cv::RNG rng;                                // OpenCV Random Number generator
    double w_sigma = 1.0;                       // Noise sigma value, w ~ N(0,σ^2)

    /* ----- Data Generation ----- */
    vector<double> x_data, y_data;              // Data Vectors

    for(int i=0; i < N; i++){
        double x = i/100.0;
        double y = exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma);  // y = exp(a.x^2+b.x+c) + w

        x_data.push_back(x);
        y_data.push_back(y);
    }

    printVec("x_data: ", x_data);
    printVec("y_data: ", y_data);

    /* ----- Build Graph for Optimization ----- */
    // The described problem is just a Curve Fitting problem, but we're using the classes types used in a Pose/Landmark Estimation problem.
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;            // The optimization variable dimension of each error term is 3, and the error value dimension is 1. template <int _PoseDim, int _LandmarkDim>
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;  // Linear Solver type

    // Gradient descent method, you can choose from GN (Gauss-Newton), LM(Levenberg-Marquardt), Powell's dog leg methods.
    g2o::OptimizationAlgorithmWithHessian *solver;

    switch (optimization_algorithm_selected){
        case 1:  // Option 1: Gauss-Newton method
            solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            break;
        case 2:  // Option 2: Levenberg-Marquardt method
            solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            break;
        case 3:  //Option 3: Powell's Dog Leg Method
            solver = new g2o::OptimizationAlgorithmDogleg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
            break;
        default:
            break;
    }

    cout << "Optimization Algorithm selected: " << optimization_algorithm_selected << endl;

    // Configure the optimizer
    g2o::SparseOptimizer optimizer;  // Graph model
    optimizer.setAlgorithm(solver);  // Set the solver
    optimizer.setVerbose(true);      // Turn on debugging output

    // Add vertices to the graph
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setId(0);                           //! Sets the id of the node in the graph, be sure that the graph keeps consistent after changing the id
    v->setEstimate(Vector3d(ae, be, ce));  //! Sets the estimate for the vertex also calls g2o::OptimizableGraph::Vertex::updateCache()
    optimizer.addVertex(v);

    // Add edges to the graph
    for (int i=0; i<N; i++){
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);  // Creates the i-th edge
        edge->setId(i);                                            // Specifies the edge ID
        edge->setVertex(0, v);                                     // Connects edge to the vertex (Node, 0)
        edge->setMeasurement(y_data[i]);                           // Observed value
        edge->setInformation(Matrix<double, 1, 1>::Identity()*(1/(w_sigma*w_sigma)));  // Information matrix: the inverse of the covariance matrix
        optimizer.addEdge(edge);
    }

    /* ----- Solve (Perform optimization) ----- */
    cout << "[g2oCurveFitting] Start optimization..." << endl;

    print("Summary: ");
    Timer t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);  // Number of optimization steps
    Timer t2 = chrono::steady_clock::now();

    printElapsedTime("Solver time: ", t1, t2);

    // Computes the RMSE
    Vector3d abc_r = {ar, br,cr};
    Vector3d abc_e = v->estimate();

    double rmse = RMSE(abc_e, abc_r);

    /* ----- Results ----- */
    cout << "---" << endl;
    cout << "Real:\t   a,b,c = ";
    cout << ar << ", " << br << ", " << cr;
    cout << endl;

    cout << "Estimated: a,b,c = ";
    cout << abc_e[0] << ", " << abc_e[1] << ", " << abc_e[2];
    cout << "\n---" <<endl;

    cout << "RMSE: " << rmse << endl;

    cout << "\nDone." << endl;

    return 0;
}
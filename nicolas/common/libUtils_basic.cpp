/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

/* Basic */
void printVec(const char text[], const std::vector<double> &vec){
    cout << text << "[";
    for(size_t i=0; i < vec.size(); i++){
        if(i != vec.size()-1){
            cout << vec.at(i) << ", ";
        }else{
            cout << vec.at(i);
        }
    }
    cout << "]" << endl << endl;
}

template <typename TTypeVec>
TTypeVec slicing(TTypeVec &arr, int begin_idx, int end_idx){
    // Starting and Ending iterators
    auto start = arr.begin() + begin_idx;
    auto end = arr.begin() + end_idx + 1;

    // To store the sliced vector
    TTypeVec result(end_idx - begin_idx + 1);

    // Copy vector using copy function()
    copy(start, end, result.begin());

    // Return the final sliced vector
    return result;
}

/* Chrono */
typedef chrono::steady_clock::time_point Timer;
void printElapsedTime(const char text[], Timer t1, Timer t2){
    chrono::duration<double> time_elapsed = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << text << time_elapsed.count() << " s" << endl;
}
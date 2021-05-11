/* =========== */
/*  Libraries  */
/* =========== */
/* System Libraries */
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

/* ============================== */
/*  Class Methods Implementation  */
/* ============================== */
void FPS::update(){
    // cout << frameCounter << endl;
    frameCounter++;
    std::time_t timeNow = std::time(0) - timeBegin;

    if (timeNow - tick >= 1){  // Print every 1s
        tick++;
        cout << "FPS: " << frameCounter << endl;
        frameCounter = 0;
    }
}
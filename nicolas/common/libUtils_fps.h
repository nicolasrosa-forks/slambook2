#ifndef LIBUTILS_FPS_H_
#define LIBUTILS_FPS_H_
// The above macro definition is to prevent redefinition errors caused by repeated references to this header file

/* ===================== */
/*  Function Prototypes  */
/* ===================== */
class FPS{
public:
    // Constructor
    FPS(){
        frameCounter = 0;
        timeBegin = std::time(0);
        tick = 0;
    }

    void update();

private:
    long frameCounter;
    std::time_t timeBegin;
    int tick;
};

#include "libUtils_fps.cpp"

#endif
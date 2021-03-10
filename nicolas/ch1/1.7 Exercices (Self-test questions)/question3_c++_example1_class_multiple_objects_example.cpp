#include <iostream> //std::cout

using namespace std;

// Create a Car class with some attributes
class Car{
    public:
        string brand;
        string model;
        int year;
};

int main(int argc, char** argv){
    // Create an object of Car
    Car carObj1;
    carObj1.brand = "BWM";
    carObj1.model = "X5";
    carObj1.year = 1999;

    // Create an object of Car
    Car carObj2;
    carObj2.brand = "Ford";
    carObj2.model = "Mustang";
    carObj2.year = 1969;

    // Print attributes values
    cout << carObj1.brand << " " << carObj1.model << " " << carObj1.year << endl;
    cout << carObj2.brand << " " << carObj2.model << " " << carObj2.year << endl;

    cout << "Done." << endl;

    return 0;
}
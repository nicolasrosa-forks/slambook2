// URL: https://www.geeksforgeeks.org/deque-cpp-stl/
#include <iostream> //std::cout
#include <deque>    //std::deque
#include <vector>   //std::vector

using namespace std;

void printDeque(std::deque<int> &var){
    std::deque<int>::iterator it;

    // Prefix ++/-- operators should be preferred for non-primitive types. 
    // Pre-increment/decrement can be more efficient than post-increment/decrement. 
    // Post-increment/decrement usually involves keeping a copy of the previous value around and adds a little extra code.
    for(it = var.begin(); it != var.end(); ++it)
        cout << *it << " " ;

    cout << endl;
}



int main(int argc, char** argv){
    //Variáveis
    std::deque<int> myDeque; // Cria um fila de ponteiros char (armazenar strings)?

    // Set some initial values
    for(int i=1; i<6; i++)
        myDeque.push_back(i);

    printDeque(myDeque);

    // Insert 10 at
    std::deque<int>::iterator it = myDeque.begin(); //it @ idx=0
    ++it; //it @ idx=1

    it = myDeque.insert(it, 10); //"it" now points to the newly inserted 10
    printDeque(myDeque);

    // Insert 2x the value 20 in the pointed position by it
    myDeque.insert(it, 2, 20);  // Cannot return a valid value to "it"!
    printDeque(myDeque);

    // Use information on the vector to add 2x 30 at third Position of the deque
    it = myDeque.begin()+2; //it @ idx=2 (Third Position)
    std::vector<int> myVector (2, 30);
    myDeque.insert(it, myVector.begin(), myVector.end());
    printDeque(myDeque);

    cout << "\nDone." << endl;

    return 0;
}
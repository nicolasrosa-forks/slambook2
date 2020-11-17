// URL: https://www.inf.pucrs.br/~pinho/PRGSWB/STL/stl.html
#include <iostream> //std::cout
#include <vector>   //std::vector
#include <algorithm> //std::sort

using namespace std;

void printVector(std::vector<float> &var){
    // Cria um iterador de inteiros
    std::vector<float>::iterator it;

    for ( it = var.begin(); it != var.end(); it++ )
        cout << "Imprimindo o vetor: " << *it << endl;

    cout << endl;
}

void isEmpty(std::vector<int> &var){
    // Testa se o vetor está vazio
    if(var.empty()){
        cout << "Vetor vazio!" << endl;
    }else{
        cout << "Vetor com elementos!" << endl;
    }
}

int main(int argc, char** argv){
    // Cria um vetor de float vazio
    std::vector<float> meuVetor;

    meuVetor.push_back(-4);
    meuVetor.push_back(4);
    meuVetor.push_back(-9);
    meuVetor.push_back(-12);
    meuVetor.push_back(40);

    printVector(meuVetor);

    sort(meuVetor.begin(), meuVetor.end());

    printVector(meuVetor);

    cout << "Done." << endl;

    return 0;
}
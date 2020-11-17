// URL: https://www.inf.pucrs.br/~pinho/PRGSWB/STL/stl.html
#include <iostream> //std::cout
#include <vector>   //std::vector

using namespace std;

void printVector(std::vector<int> &var){
    for (int i=0; i<var.size(); i++)
        cout << "Imprimindo vetor: " << var[i] << endl;
    
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
    // Cria um vetor de inteiros vazio
    std::vector<int> meuVetor;

    isEmpty(meuVetor);

    // Inclue no fim do vetor um elemento
    meuVetor.push_back(7);
    meuVetor.push_back(11);
    meuVetor.push_back(2006);

    isEmpty(meuVetor);

    // Printa conteúdo do vetor, imprimirá três valores {7, 11, 2006}
    printVector(meuVetor);

    // Retira o último elemento
    meuVetor.pop_back();
    
    // Printa conteúdo do vetor, imprimirá 2 valores {7, 11}
    printVector(meuVetor);

    cout << "Done." << endl;

    return 0;
}
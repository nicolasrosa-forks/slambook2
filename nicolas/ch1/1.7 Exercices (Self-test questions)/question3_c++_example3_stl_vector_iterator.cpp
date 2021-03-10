// URL: https://www.inf.pucrs.br/~pinho/PRGSWB/STL/stl.html
#include <iostream> //std::cout
#include <vector>   //std::vector

using namespace std;

void printVector(std::vector<int> &var){
    // Cria um iterador de inteiros
    std::vector<int>::iterator it;

    for(it = var.begin(); it != var.end(); it++ )
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

    // Insere 55 como segundo elemento, deslocando os demais para a próxima posição
    meuVetor.insert(meuVetor.begin() + 1, 55);

    // Printa conteúdo do vetor, imprimirá quatro valores {7, 55, 11, 2006}
    printVector(meuVetor);

    // Retira 11 da lista (terceira posição)
    meuVetor.erase(meuVetor.begin() + 2);

    // Agora, tem que imprimir três de novo {7, 55, 2006}
    printVector(meuVetor);

    // Limpa todo o vetor
    isEmpty(meuVetor);

    meuVetor.clear();

    isEmpty(meuVetor);

    cout << "Done." << endl;

    return 0;
}
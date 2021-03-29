// URL: https://www.inf.pucrs.br/~pinho/PRGSWB/STL/stl.html
#include <iostream> //std::cout
#include <vector>   //std::vector
#include <algorithm> //std::sort

using namespace std;

class Pessoa{
    // Private
    string nome;
    int idade;

    public:
        // Constructor
        Pessoa(const string &no, int id){
            nome = no;
            idade = id;
        }
        string getNome(){
            return nome;
        }
        int getIdade(){
            return idade;
        }
};

void printVector(std::vector<Pessoa> &var){
    std::vector<Pessoa>::iterator it; //ptr

    // Prefix ++/-- operators should be preferred for non-primitive types. 
    // Pre-increment/decrement can be more efficient than post-increment/decrement. 
    // Post-increment/decrement usually involves keeping a copy of the previous value around and adds a little extra code.
    for (it = var.begin(); it != var.end(); ++it){
        cout << "Nome: " << it->getNome();
        cout << "\tIdade: " << it->getIdade() << endl;
    }

    cout << endl;
}

bool ordena_por_nome(Pessoa A, Pessoa B){
    if (A.getNome() < B.getNome()) // Ordem Alfabetica
        return true;

    return false;
}

bool ordena_por_idade(Pessoa A, Pessoa B){
    if (A.getIdade() > B.getIdade()) // Ordem Crescente
        return true;

    return false;
}


int main(int argc, char** argv){
    std::vector<Pessoa> Pessoas; // Vector de Pessoas

    Pessoas.push_back(Pessoa("Joao", 25));
    Pessoas.push_back(Pessoa("Maria", 32));
    Pessoas.push_back(Pessoa("Carla", 4));
    Pessoas.push_back(Pessoa("Abel", 30));

    cout << "Ordenado conforme foi preenchido:" << endl;
    printVector(Pessoas);

    // Ordena por Nome
    sort(Pessoas.begin(), Pessoas.end(), ordena_por_nome);
    cout << "Ordenado por Nome:" << endl;
    printVector(Pessoas);

    // Ordena por Idade
    sort(Pessoas.begin(), Pessoas.end(), ordena_por_idade);
    cout << "Ordenado por Idade:" << endl;
    printVector(Pessoas);

    cout << "Done." << endl;

    return 0;
}
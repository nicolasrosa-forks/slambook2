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
        Pessoa(string no, int id){
            idade = id;
            nome = no;
        }
        string getNome(){
            return nome;
        }
        int getIdade(){
            return idade;
        }
};

void printVector(std::vector<Pessoa> &var){
    std::vector<Pessoa>::iterator it;

    for (it = var.begin(); it != var.end(); it++){
        cout << "Nome: " << it->getNome();
        cout << "\tIdade: " << it->getIdade() << endl;
    }

    cout << endl;
}


int main(int argc, char** argv){
    std::vector<Pessoa> Pessoas; // Vector de Pessoas
    
    Pessoas.push_back(Pessoa("Joao", 25));
    Pessoas.push_back(Pessoa("Maria", 32));
    Pessoas.push_back(Pessoa("Carla", 4));
    Pessoas.push_back(Pessoa("Abel", 30));

    printVector(Pessoas);

    cout << "Done." << endl;

    return 0;
}
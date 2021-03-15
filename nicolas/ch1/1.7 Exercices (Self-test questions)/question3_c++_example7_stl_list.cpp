// URL: https://www.inf.pucrs.br/~pinho/PRGSWB/STL/stl.html
#include <iostream>  //std::cout
#include <list>      //std::list
#include <algorithm> //std::sort

using namespace std;

void printList(std::list<double> &var){
    std::list<double>::iterator it; //Cria um iterador de float

    // Prefix ++/-- operators should be preferred for non-primitive types. 
    // Pre-increment/decrement can be more efficient than post-increment/decrement. 
    // Post-increment/decrement usually involves keeping a copy of the previous value around and adds a little extra code.
    for(it=var.begin();it != var.end(); ++it)
        cout << "Imprimindo a lista: " << *it << endl;

    cout << endl;

}

int main(int argc, char** argv){
    //Variáveis
    std::list<double> minhaLista;  // Cria uma lista de floats vazia
    std::list<double>::iterator k; //Cria um iterador de float

    minhaLista.push_back(7.5);
    minhaLista.push_back(27.26);
    minhaLista.push_front(-44); // Inserindo no início da lista
    minhaLista.push_front(7.5); // Inserindo no início da lista
    minhaLista.push_back(69.09);

    printList(minhaLista);

    // Insere -2.888 como último elemento
    minhaLista.insert(minhaLista.end(), -2.888);
    printList(minhaLista);

    // Retira o elemento -44 da lista
    minhaLista.remove(-44);
    printList(minhaLista);

    // Remove elementos duplicados da lista (no caso, 7.5 aparece 2x)
    minhaLista.unique();
    printList(minhaLista);

    // Ordena a lista, em ordem ascendente
    minhaLista.sort();
    printList(minhaLista);

    // Para usar find, informe  o ponto inicial e final de procura, mais o elemento
    // este método STL devolve um iterador (ponteiro) para o objeto.
    k = find(minhaLista.begin(), minhaLista.end(), 27.26);

    if(*k == 27.26)
        cout << "Elemento 27.26 encontrado!!!" << endl;
    else
        cout << "Não existe o elemento procurado!!!" << endl;

    cout << endl;

    // Limpa toda a lista
    minhaLista.clear();

    cout << "Done." << endl;

    return 0;
}
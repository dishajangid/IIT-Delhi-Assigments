// Do NOT add any other includes
#include <string> 
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

class dNode{
public:

    char letter;
    int count;

    vector<dNode*> children;

    dNode(char let);

    ~dNode();

};

class Dict {
private:
    // You can add attributes/helper functions here

    dNode* root;

public: 
    /* Please do not touch the attributes and 
    functions within the guard lines placed below  */
    /* ------------------------------------------- */
    Dict();

    ~Dict();

    void insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence);

    long long get_word_count(string word);

    void dump_dictionary(string filename);

    /* -----------------------------------------*/

    void insert(string s);
};
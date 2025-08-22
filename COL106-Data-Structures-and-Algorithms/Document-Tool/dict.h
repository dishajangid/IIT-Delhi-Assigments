// Do NOT add any other includes
#include <string> 
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

class Dict {
private:
    // You can add attributes/helper functions here
    struct KeyValue{
        string key;
        int value;
        KeyValue(const string& k , int v){
            key = k;
            value = v;
        }
    };
    vector<vector<KeyValue>> wordCount;

    //Helper functions

    vector<string> tokenize(const string& sentence);
    int hashPrime(const string& word);
    int findWordIndex(int hashValue , const string& word);
public: 
    /* Please do not touch the attributes and 
    functions within the guard lines placed below  */
    /* ------------------------------------------- */
    Dict();

    ~Dict();

    void insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence);

    int get_word_count(string word);

    void dump_dictionary(string filename);

    /* -----------------------------------------*/
};
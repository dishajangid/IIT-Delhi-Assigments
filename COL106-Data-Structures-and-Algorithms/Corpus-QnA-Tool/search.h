// Do NOT add any other includes
#include <string> 
#include <vector>
#include <iostream>
#include "Node.h"
using namespace std;


class SearchEngine {
private:
    // You can add attributes/helper functions here
    vector<int> preprocessSentence(const string& sentence);
    bool matchPattern(const vector<int>& sentenceNumbers , const vector<int>& patternNumbers , pair<vector<int> , Node*> const& data);

    vector<pair<vector<int> , Node*>> sentencedata;
    int hashFunction(const char& word);


public: 
    /* Please do not touch the attributes and 
    functions within the guard lines placed below  */
   

    /* ------------------------------------------- */
    SearchEngine();

    ~SearchEngine();

    void insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence);

    void search(string pattern, int& n_matches);

    /* -----------------------------------------*/
    
    
};
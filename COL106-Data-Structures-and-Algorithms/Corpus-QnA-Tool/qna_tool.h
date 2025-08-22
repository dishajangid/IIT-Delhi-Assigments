#include <iostream>
#include <fstream>
#include "Node.h"
#include <vector>
#include "dict.h"
#include "search.h"

using namespace std;

//Trie implementation
struct TrieNode{
    public:
        vector<TrieNode*> children;
        bool isEndOfWord;
        TrieNode():isEndOfWord(false){
            children = vector<TrieNode*> (26,nullptr);
        }
};
class Trie{
        TrieNode* root;
        public:
            Trie(){
                root = new TrieNode();
            }
            void insert(const string& word){
                TrieNode* current = root;
                for(char c: word){
                    if(!current->children[c-'a']){
                        current->children[c-'a'] = new TrieNode();
                    }
                    current = current->children[c-'a'];
                }
                current->isEndOfWord = true;
            }
            bool search(const string& word){
                TrieNode* current = root;
                for(char c: word){
                    if(!current->children[c-'a']){
                        return false;
                    }
                    current = current->children[c-'a'];
                }
                return current->isEndOfWord;
            }       
};

//hashFunction to store the unwanted words
// class hashTrie{
//     vector<Trie*> hashTable;
//     public:
//         hashTrie(int tableSize):hashTable(tableSize , nullptr){};
//         int hashFunction(const string& word){
//             return word.length() % hashTable.size();
//         }
//         void insert(const string& word){
//             int index = hashFunction(word);
//             if(!hashTable[index]){
//                 hashTable[index] = new Trie();
//             }
//             hashTable[index]->insert(word);
//         }
//         bool search(const string& word){
//             int index = hashFunction(word);
//             if(hashTable[index]){
//                 return hashTable[index]->search(word);
//             }
//             return false;
//         }
// };

 struct Sentence{
    int mybookcode;
    int mypage;
    int myparagraph;
    int mysentencenumber;
    string mysentence;
};
 

class QNA_tool {
private:

    // You are free to change the implementation of this function
    void query_llm(string filename, Node* root, int k, string API_KEY, string question);

    // You can add attributes/helper functions here

public:

    /* Please do not touch the attributes and
    functions within the guard lines placed below  */
    /* ------------------------------------------- */
    
    QNA_tool(); // Constructor
    ~QNA_tool(); // Destructor
    Trie* trie;

    void insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence);
    // This function is similar to the functions in dict and search 
    // The corpus will be provided to you via this function
    // It will be called for each sentence in the corpus

    Node* get_top_k_para(string question, int k);
    // This function takes in a question, preprocess it
    // And returns a list of paragraphs which contain the question
    // In each Node, you must set: book_code, page, paragraph (other parameters won't be checked)
    std::string get_paragraph(int book_code, int page, int paragraph);
    // Given the book_code, page number, and the paragraph number, returns the string paragraph.
    // Searches through the corpus.
    void query(string question, string filename);
    // This function takes in a question and a filename.
    // It should write the final answer to the specified filename.

    /* -----------------------------------------*/
    /* Please do not touch the code above this line */

    // You can add attributes/helper functions here
    //string get_paragraph(int book_code, int page, int paragraph);

    //vector<vector<string>>para_store;

    Dict* dict;
    SearchEngine* se;
    
    vector<string> preProcess(const string& sentence);
    void insertInTrie(const string& filename);

    vector<Sentence> sentences;

};
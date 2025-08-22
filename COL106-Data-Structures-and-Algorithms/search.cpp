// Do NOT add any other includes
#include "search.h"
#include <functional>

SearchEngine::SearchEngine(){
    // Implement your function here  
    sentencedata.resize(1000000);
}

SearchEngine::~SearchEngine(){
    // Implement your function here  
    for(const auto& data : sentencedata){
        if(data.second){ delete data.second;}
    }

}
int SearchEngine::hashFunction(const char& word){       //hashFunction
    int hashValue = 5;
    hashValue = ((hashValue << 1) + hashValue)^word;
    return hashValue;
}
vector<int> SearchEngine::preprocessSentence(const string& sentence){
    vector<int> wordNumbers;
    char word;
    const std::string delimiters = " .-:!()\"""''?_[]\"';@, ";
    for(char c: sentence){
        if(isalnum(c)){          //now it is storing char
            word=tolower(c);
            wordNumbers.push_back(hashFunction(word));
        }else if(c == ' '){
            word = c;
            wordNumbers.push_back(hashFunction(word));
        }else if(delimiters.find(c) == std::string::npos){
            wordNumbers.push_back(hashFunction(c));
        }else{
            wordNumbers.push_back(hashFunction(' '));
        }
    }
    //     else if(word != '\0'){
    //         int wordNumber = hashFunction(word); 
    //         wordNumbers.push_back(wordNumber);
    //         word = '\0';
    //     }
    // }
    // if(word != '\0'){
    //     int wordNumber = hashFunction(word);
    //     wordNumbers.push_back(wordNumber);
    //     word = '\0';
    // }
    return wordNumbers;
}
void SearchEngine::insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence){
    Node* node = new Node(book_code , page , paragraph , sentence_no , -1); //what is offset
    vector<int> wordNumbers = preprocessSentence(sentence);
    sentencedata.push_back({wordNumbers  , node});

    
    
}
bool SearchEngine::matchPattern(const vector<int>& sentenceNumbers , const vector<int>& patternNumbers , pair<vector<int> , Node*> const& data){
    if(patternNumbers.size() > sentenceNumbers.size()){
        return false;
    }
    for(int i = 0; i <= (sentenceNumbers.size() - patternNumbers.size()) ; i++){
        bool match = true;
        for(int j = 0; j < patternNumbers.size() ; j++){
            if(sentenceNumbers[i+j] != patternNumbers[j]){
                match = false;
                break;
            }

        }if(match){
            data.second->offset = i; //changing offset
            return true;
        }
    }
    
    return false;
}

void SearchEngine::search(string pattern, int& n_matches){
    vector<int> patternNumbers = preprocessSentence(pattern);
    
    n_matches = 0;
    Node* resultHead = nullptr;
    for(const auto& data : sentencedata){
        if(matchPattern(data.first , patternNumbers , data)){
            n_matches++;
            /*if(!resultHead){
                resultHead = data.second;
                //resultHead = new Node(data.second->book_code , data.second->page , 
                //          data.second->paragraph , data.second->sentence_no , data.second->offset);
            }else{
                Node* newNode = data.second;
                // resultHead->right = newNode;  //growing the linked list to right;
                // newNode->left = resultHead;
               
                // if(resultHead->right == nullptr){
                //     resultHead->right = newNode;
                //     newNode->left = resultHead;
                // }
                // if(resultHead->right != nullptr){
                //     Node* temp = resultHead;
                //     while(temp->right != nullptr){

                //         temp = temp->right;
                //     }
                //     temp->right  = newNode;
                //     newNode->left = temp;
                // }

                Node* temp = resultHead;
                while(temp->right != nullptr){
                    temp = temp->right;
                }
                temp->right = newNode;
                newNode->left = temp;
            }*/
        }
    }
    return;
}

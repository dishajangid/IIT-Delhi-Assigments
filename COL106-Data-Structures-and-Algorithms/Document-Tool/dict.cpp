// Do NOT add any other includes
#include "dict.h"
#include <cctype>
#include <fstream>
#include <string>

Dict::Dict(){
    // Implement your function here 
    wordCount.resize(404960); 
}

// vector<string> Dict:: tokenize(const string& sentence){
//     vector<string> words;
//     string word;

//     for(char c : sentence){
//         if(isalnum(c)){
//             word += tolower(c);
//         }
//         else if(!word.empty()){
//             words.push_back(word);
//             word.clear();
//         }else if(c != ' '){
//             words.push_back(to_string(c));
//         }
//     }
//     if(!word.empty()){
//         words.push_back(word);
//         word.clear();
//     }
//     return words;
// }
vector<string> Dict::tokenize(const string& sentence){
    vector<string> words;
    string word;
    const std::string delimiters = " .-:!()""''`\"\'?[]\"\';@,";
    for(char c: sentence){
        if(delimiters.find(c) == std::string::npos){ //if not in delimiters
            // if(isalnum(c)){word += tolower(c);}
            // else{
            //     word+=c;
            // }
            word+=tolower(c);
        }else{
            if(!word.empty()){
                words.push_back(word);
                //cout<<word<<" ";
                word.clear();
            }
        }
    }
    if(!word.empty()){
        words.push_back(word);
        //word.clear();
    }
    return words;
}


int Dict::hashPrime(const string& word){
    
    int hashValue = 0;
    const int prime = 13;
    for(char c: word){
        //hashValue = ((hashValue<<1) + hashValue)^ (c - 'a');
        c = tolower(c);
        hashValue = (hashValue)^(c);
    }
    
    if(hashValue < 0){
        hashValue = -hashValue;
    }
    return hashValue%404960;
}

int Dict::findWordIndex(int hashValue , const string& word){
    string x;
    for(char c:word){
        x+=tolower(c);
    }
    if(wordCount[hashValue].size() == 0){return -1;} //word not found

    for(int i=0; i < (wordCount[hashValue]).size(); i++){
        if((wordCount[hashValue][i]).key == x){
            //cout<<i<<endl;
            return i; //index of the word in wordCount[hashValue]
        }
    }
    return -1; //word not found
}

Dict::~Dict(){
    // Implement your function here  
    //dump_dictionary("dictionary.txt"); 
}

void Dict::insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence){
    // Implement your function here  
    vector<string> sentenceWords = tokenize(sentence);
    // for(auto& word: sentenceWords){
    //     cout<<word<<" ";
    // }
    // cout<<endl;
    
    for(const string& word : sentenceWords){
        int hashValue = Dict::hashPrime(word);
        //cout<<word<<"  hashValue = "<<hashValue<<endl;
        int index = findWordIndex(hashValue , word);
        //cout<<word<<" "<<index<<endl;
        if(index != -1){
            // if(word == "11/2s"){
            //     cout<<"got it   " << wordCount[hashValue][index].value + 1<<endl;
            // }
            (wordCount[hashValue][index].value)+=1;
            //cout<<wordCount[hashValue][index].value<<endl;
        }else{
            // if(word.find('/') != string::npos || isdigit(word[0]) || word.find('&') != string::npos)
            //     wordCount[hashValue].push_back((KeyValue(word,1)));
            // else
                wordCount[hashValue].push_back((KeyValue(word,1)));

        }
    }
}

int Dict::get_word_count(string word){
    //if(word == "&"){cout<<"yesssssssssss"<<endl;}
    // Implement your function here 
    string x;
    const std::string delimiters = " .-:!()""''?,`[]\"\';@,";
    for(char c:word){
        //cout<<"c is in get_word count  "<<c<<endl;
        if(delimiters.find(c) == std::string::npos){
            if(isalnum(c))
                x+=tolower(c);
            else{
                x+=c;
            }
        }  
    }
    word = x;
    int hashValue = Dict::hashPrime(word);
    
    int index = findWordIndex(hashValue , word);
    if(index != -1){
        //cout<<word <<" "<<wordCount[hashValue][index].value<<endl;;
        return wordCount[hashValue][index].value;
    }else{
        return 0;
    }
}

void Dict::dump_dictionary(string filename){
    // Implement your function here  
    ofstream file(filename);
    if(file.is_open()){
        for(int j = 0; j < wordCount.size(); j++){
            for(int i = 0; i < (wordCount[j]).size(); i++){
                //cout<<  wordCount[j][i].key << ", " << wordCount[j][i].value << endl;
                file << wordCount[j][i].key << ", " << wordCount[j][i].value << endl;
                //cout<< wordCount[j][i].key << ", " << wordCount[j][i].value << endl;
            }
        }
        file.close();
    }else{                                               
        cerr << "Unable to open file for writing" << endl;
    }
}





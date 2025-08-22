// Do NOT add any other includes
#include "dict.h"

void dfs(dNode* n,string p,ofstream& fil){

    if(n->letter != '.'){p.push_back(n->letter);}
    
    int k = n->children.size();

    if(k!=0){
    
    for(int i = 0;i<54;i++){
        if(n->children[i] != nullptr){
        dfs(n->children[i],p,fil);
        }
    }

    }

    if(n->count != 0){  
  fil<<p<<", "<<n->count<<endl;
   }

}

string decap(string s){
    string d = s;
    int k = s.length();

    for(int i = 0;i<k;i++){
        if(int(s[i])<=90 && int(s[i] >=65)){int c = int(s[i])+32;d[i] = char(c);}
    }

    return d;
}

vector<string> parser(string s){
    vector<string> ans = {};
    string d = s;
    int k = s.length();
    int i = 0;
    d.push_back('.');
    string temp= "";
    while(i<=k){//int q = int(d[i]);
        if(d[i] == ' ' || d[i] == '\"' || d[i] == '.' || d[i] == ',' || d[i] == '-' || d[i] == ':' || d[i] == '!' || d[i] == '(' || d[i] == ')' || d[i] == '?' || d[i] == '[' || d[i] == ']' || d[i] == '\'' || d[i] == ';' || d[i] == '@' || d[i] == '—' || d[i] == '˙' || d[i] == '”' || d[i] == '“' || d[i] == '’' || d[i] == '‘' ){
            //if(q<48 || (q<65&& q>57) || (q>90 && q<=96) || q>=123){
            if(temp != ""){ans.push_back(temp);
            temp = "";}

            else{
                temp = "";
            }
        }

        else{
            temp.push_back(d[i]);
        }

        i++;
    }

    int k1 = ans.size();

    for(int i = 0;i<k1;i++){
        ans[i] = decap(ans[i]);
    }

    return ans;


}

dNode::dNode(char let){
    letter = let;
    count = 0;
    children = {};
   
    }

dNode::~dNode(){
int k = children.size();

    for(int i =0;i<k;i++){
        if(children[i] != nullptr){
        delete children[i];
        }
    }




}


Dict::Dict(){
    // Implement your function here 
    root = new dNode('.');   
    // for(int i = 97;i<123;i++){
    //     dNode* n = new dNode(char(i));
    //     root->children[i-97] = n;

    // }


}

Dict::~Dict(){
    // Implement your function here   
    delete root; 
}

void Dict::insert(string s){
    int k = s.length();
    int i = 0;
    dNode* n = root;

    while(i<k){

        //if(i == 0){

            //int l = n->children.size();
            // if(l ==0){
            //   for(int i = 0;i<26;i++){
            //   n->children.push_back(NULL);
            //   } 
            // }

            int h;

            if(s[i] == '_'){h = 36;}

            else if(s[i] == '&'){h = 37;}
            else if(s[i] == '/'){h = 38;}
            else if(s[i] == '\\'){h = 39;}
            else if(s[i] == '%'){h = 40;}
            else if(s[i] == '#'){h = 41;}
            else if(s[i] == '$'){h = 42;}
            else if(s[i] == '<'){h = 43;}
            else if(s[i] == '>'){h = 44;}
            else if(s[i] == '^'){h = 45;}
            else if(s[i] == '`'){h = 46;}
            else if(s[i] == '{'){h = 47;}
            else if(s[i] == '}'){h = 48;}
            else if(s[i] == '|'){h = 49;}
            else if(s[i] == '='){h = 50;}
            else if(s[i] == '~'){h = 51;}
            else if(s[i] == '*'){h = 52;}
            else if(s[i] == '+'){h = 53;}

                else if(int(s[i]) >=97){

            h = int(s[i]) - 97;}

            else{
                h = int(s[i]) -22;
            }
           // }

           // cout<<s[i]<<endl;

            if(n->children.size() == 0){
                
   // for(int i = 0;i<52;i++){
         //dNode* l = new dNode('-');
    n->children = vector<dNode*>(54,nullptr);
   // }
 
            }

//cout<<"hi"<<endl;

            if(n->children[h] == nullptr){

            

                dNode* t = new dNode(s[i]);
                //delete n->children[h];
                n->children[h] = t;

            }
            n = n->children[h];

        i++;

        }
        


    n->count += 1;



}

void Dict::insert_sentence(int book_code, int page, int paragraph, int sentence_no, string sentence){
    // Implement your function here 
    vector<string> a = parser(sentence); 
    int k = a.size();
    
    for(int i = 0;i<k;i++){
        
        // if(a[i] == "a00"){
        // cout<<a[i]<<endl;}
        insert(a[i]);

    }
    return;
}

long long Dict::get_word_count(string word){
    // Implement your function here  

    int k = word.length();
    int i = 0;
    dNode* n = root;

    while(i<k){

        int l = n->children.size();
        if(l == 0){
             return 0;
         }

            // int h;
            // if(int(word[i])<=90){h = int(word[i]) - 65;}
            // else{
            int h;

            if(word[i] == '_'){h = 36;}

            else if(word[i] == '&'){h = 37;}
            else if(word[i] == '/'){h = 38;}
            else if(word[i] == '\\'){h = 39;}
            else if(word[i] == '%'){h = 40;}
            else if(word[i] == '#'){h = 41;}
            else if(word[i] == '$'){h = 42;}
            else if(word[i] == '<'){h = 43;}
            else if(word[i] == '>'){h = 44;}
            else if(word[i] == '^'){h = 45;}
            else if(word[i] == '`'){h = 46;}
            else if(word[i] == '{'){h = 47;}
            else if(word[i] == '}'){h = 48;}
            else if(word[i] == '|'){h = 49;}
            else if(word[i] == '='){h = 50;}
            else if(word[i] == '~'){h = 51;}
            else if(word[i] == '*'){h = 52;}
            else if(word[i] == '+'){h = 53;}

                else if(int(word[i]) >=97){

            h = int(word[i]) - 97;}

            else{
                h = int(word[i]) -22;
            }
            
           // }

            if(n->children[h] == nullptr){return 0;}
            n = n->children[h];

        i++;

    }

    return n->count;
    
    
}

void Dict::dump_dictionary(string filename){
    // Implement your function here  
     ofstream fil;
    fil.open(filename, ios_base::out | ios_base::app);
    dfs(root,"",fil);

    fil.close();
    return;
}
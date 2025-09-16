#include <iostream>
#include <vector>

// Iterator interface
class Iterator {
public:
    virtual ~Iterator() = default;
    virtual bool hasNext() = 0;
    virtual int next() = 0;
};

// Concrete Iterator
class VectorIterator : public Iterator {
    std::vector<int>& data_;
    size_t index_ = 0;
public:
    explicit VectorIterator(std::vector<int>& d) : data_(d) {}
    bool hasNext() override { return index_ < data_.size(); }
    int next() override { return data_[index_++]; }
};

// Aggregate interface
class Aggregate {
public:
    virtual ~Aggregate() = default;
    virtual Iterator* createIterator() = 0;
};

// Concrete Aggregate
class IntCollection : public Aggregate {
    std::vector<int> data_;
public:
    void add(int x) { data_.push_back(x); }
    Iterator* createIterator() override {
        return new VectorIterator(data_);
    }
};

// Client
int main() {
    IntCollection col;
    col.add(10);
    col.add(20);
    col.add(30);

    Iterator* it = col.createIterator();
    while (it->hasNext()) {
        std::cout << it->next() << " ";
    }
    std::cout << "\n";

    delete it; // clean up
}


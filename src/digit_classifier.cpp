#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include "stb_image.h"
#include <Eigen/Dense>
#include "utec/nn/neural_network.h"

using namespace utec;
using namespace std;

static algebra::Tensor load_csv(const string& path, algebra::Tensor* labels=nullptr) {
    ifstream file(path);
    if (!file.is_open()) throw runtime_error("Cannot open file " + path);
    string line;
    vector<vector<float>> data;
    vector<int> lbs;
    while (getline(file, line)) {
        stringstream ss(line);
        string item;
        vector<float> row;
        bool first=true;
        int label=0;
        while (getline(ss, item, ',')) {
            if (first && labels) {
                label = stoi(item);
                lbs.push_back(label);
                first=false;
            } else {
                row.push_back(stof(item)/255.0f);
            }
        }
        if (!labels) row.insert(row.begin(), label); // if labels==null maybe for features only? But we only call with labels
        data.push_back(row);
    }
    size_t rows=data.size();
    size_t cols=data[0].size();
    algebra::Tensor X(rows, cols);
    for (size_t i=0;i<rows;++i)
        for (size_t j=0;j<cols;++j)
            X(i,j)=data[i][j];
    if (labels) {
        size_t c = *max_element(lbs.begin(), lbs.end())+1;
        *labels = algebra::Tensor::Zero(rows, c);
        for (size_t i=0;i<rows;++i)
            (*labels)(i, lbs[i]) = 1.0f;
    }
    return X;
}

static algebra::Tensor load_image(const string& path) {
    int w,h,n;
    unsigned char* data = stbi_load(path.c_str(), &w,&h,&n, 1);
    if (!data) throw runtime_error("Cannot load image");
    algebra::Tensor X(1, w*h);
    for (int i=0;i<w*h;++i) {
        X(0,i) = data[i]/255.0f;
    }
    stbi_image_free(data);
    return X;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " train CSV|IMAGE [IMAGE..]" << endl;
        return 1;
    }
    string cmd = argv[1];
    if (cmd == "train") {
        if (argc < 3) {
            cerr << "Provide path to mnist CSV" << endl;
            return 1;
        }
        algebra::Tensor labels;
        algebra::Tensor X = load_csv(argv[2], &labels);
        nn::NeuralNetwork net;
        net.train(X, labels, 10, 0.5f);
        filesystem::create_directories("models");
        net.save("models/digit_weights.bin");
        cout << "Model saved to models/digit_weights.bin" << endl;
        return 0;
    } else {
        nn::NeuralNetwork net;
        net.load("models/digit_weights.bin");
        for (int i=1;i<argc;++i) {
            auto img = load_image(argv[i]);
            auto prob = net.predict(img);
            Eigen::Index idx;
            prob.row(0).maxCoeff(&idx);
            cout << argv[i] << ": " << idx << endl;
        }
    }
    return 0;
}


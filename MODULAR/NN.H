#ifndef NEURAL_H
#define NEURAL_H

#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <ctime>
#include <cmath>

#include "LAYER.H"

class BasicLayer : public Layer {
    public:
    std::vector<std::vector<double>> weights; // Again, weights[i][j] is the scale of the ith input to the jth output
    
    BasicLayer() {
        in_m = out_m = 1;
        in_n = 1;
        out_n = 1;
        
        weights = std::vector<std::vector<double>>(in_n + 1, std::vector<double>(out_n, 1));
    }
    
    BasicLayer(int in, int out) {
        in_m = out_m = 1;
        in_n = in;
        out_n = out;
        
        weights = std::vector<std::vector<double>>(in_n + 1, std::vector<double>(out_n, 1));
    }
    
    BasicLayer(std::vector<std::vector<double>> w) {
        in_m = out_m = 1;
        in_n = w.size() - 1;
        out_n = w[0].size();
        
        weights = std::vector<std::vector<double>>(w);
    }
    
    BasicLayer(BasicLayer& other) {
        in_n = other.in_n;
        out_n = other.out_n;
        weights = std::vector<std::vector<double>>(other.weights);
    }
    
    std::vector<std::vector<double>> compute(std::vector<std::vector<double>> input) {
        std::vector<std::vector<double>> output(out_n, std::vector<double>(out_m, 0));
        for (int i = 0; i < out_n; i++) {
            for (int j = 0; j < in_n; j++) {
                // std::cout << i << " " << j << std::endl;
                output[i][0] += input[j][0] * weights[j][i];
            }
            output[i][0] += weights[in_n][i];
        }
        return output;
    }
    
    // If Y[i] = SUM(w[j][i] * X[j]) + B then dE/d(X[j]) = sum(i) dE/dY[i] w[j][i] and dE/d(w[j][i]) = dE/dY[i] X[j] and dE/dB = dE/dY same as the CNN
        
    std::vector<std::vector<double>> weightgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        std::vector<std::vector<double>> res(weights.size() - 1, std::vector<double>(weights[0].size()));
        for (int i = 0; i < res.size(); i++) {
            for (int j = 0; j < res[i].size(); j++) res[i][j] = nextlayergradient[j][0] * inputs[i][0];
        }
        return res;
    }
    std::vector<std::vector<double>> biasgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        return nextlayergradient;
    }
    
    std::vector<std::vector<double>> elementgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        // Nextlayergradient alreayd takes care of the dE/dY for us.
        std::vector<std::vector<double>> res(in_n, std::vector<double>(1, 0));
        
        for (int i = 0; i < in_n; i++) {
            for (int j = 0; j < out_n; j++) res[i][0] += nextlayergradient[j][0] * weights[i][j];
        }
        return res;
    }
    
    std::vector<std::vector<double>> backprop(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs, double alpha = 0.01, bool VERBOSE = false) {
        auto eg = elementgrads(nextlayergradient, inputs);
        auto bg = biasgrads(nextlayergradient, inputs);
        auto wg = weightgrads(nextlayergradient, inputs);
        
        for (int i = 0; i < bg.size(); i++) weights[in_n][i] -= alpha * bg[i][0];
        for (int i = 0; i < in_n; i++) {
            for (int j = 0; j < out_n; j++) weights[i][j] -= wg[i][j] * alpha;
        }
        return eg;
    }
    
    std::string toString() {
        std::string res = "INPUT [" + std::to_string(in_n) + " " + std::to_string(in_m) + "] OUTPUT [" + std::to_string(out_n) + " " + std::to_string(out_m) + "]\n";
        res = res + vtos(weights);
        return res;
    }
};

#endif

/*

EXAMPLE CODE



#include "LAYER.H"
#include "NN.H"

#include <bits/stdc++.h>
using namespace std;

int N = 1000;
int M = 100;

void nntest(bool VERBOSE = false, vector<vector<double>> centers = {{8, 4}, {-10, -8}}, vector<double> radii = {8, 6}) {
    BasicLayer layer1(Layer::random(centers[0].size() + 1, 10, 1));
    SigmoidLayer act1(10, 10);
    BasicLayer layer2(Layer::random(11, centers.size(), 1));
    SigmoidLayer act2(centers.size(), centers.size());
    
    cout << layer1.toString();
    
    int good = 0;
    int bad = 0;
    
    for (int iter = 1; iter <= N; iter++) {
        bool MS = (iter % (N / M) == 0);
        
        auto v = Layer::random(2, 1, 16);
        auto sum1 = layer1.compute(v);
        auto sig1 = act1.compute(sum1);
        auto sum2 = layer2.compute(sig1);
        auto out = act2.compute(sum2);
        
        std::vector<std::vector<double>> desired = Layer::constant(centers.size(), 1, -1);
        for (int i = 0; i < centers.size(); i++) {
            double test = 0;
            for (int idx = 0; idx < v.size(); idx++) test += (v[idx][0] - centers[i][idx]) * (v[idx][0] - centers[i][idx]);
            if (test <= radii[i] * radii[i]) desired[i][0] = 1;
        }
        
        for (int i = 0; i < desired.size(); i++) {
            if (abs(desired[i][0] - out[i][0]) < abs(desired[i][0] + out[i][0])) good++;
            else bad++;
        }
        
        auto error = Layer::diff(out, desired);
        error = act2.backprop(error, sum2, 0.01);
        error = layer2.backprop(error, sig1, 0.01);
        error = act1.backprop(error, sum1, 0.01);
        layer1.backprop(error, v, 0.01);
        
        if (MS) {
            cout << good << " PASS " << bad << " FAIL " << endl;
            good = 0;
            bad = 0;
        }
    }
}

int main()
{
    printf("Hello World");
    
    nntest(false);

    return 0;
}

*/

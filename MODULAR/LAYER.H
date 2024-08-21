#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <ctime>
#include <cmath>

// Simple implementation of a convolutional neural network. Convolutions do not take elements from outside the input arrays.
// Based on this article https://medium.com/@kattarajesh2001/convolutional-neural-network-from-scratch-0d7513d62923
// This one is more modular -- instead of the entire NN being a class we have classes for layers. You will have to arrange them into the CNN.

// A layer takes in a (in_n)x(in_m) vector and returns an (out_n)x(out_m) vector.
class Layer {
    public:
    int out_n, out_m;
    int in_n, in_m;
    
    // METHODS THAT DIFFER BETWEEN CLASSES
    
    Layer() {
        out_n = 1;
        out_m = 1;
        in_n = 1;
        in_m = 1;
    }
    
    Layer(int a, int b, int ia, int ib) {
        out_n = a;
        out_m = b;
        in_n = ia;
        in_m = ib;
    }
    
    Layer(const Layer& other) {
        out_n = other.out_n;
        out_m = other.out_m;
        in_n = other.in_n;
        in_m = other.in_m;
    }
    
    // Forward and backward passing (THESE CHANGE ACROSS LAYERS)
    
    std::vector<std::vector<double>> compute(std::vector<std::vector<double>> input) {
        std::vector<std::vector<double>> output(out_n, std::vector<double>(out_m, 0));
        for (int i = 0; i < in_n && i < out_n; i++) {
            for (int j = 0; j < in_m && j < out_m; j++) output[i][j] = input[i][j];
        }
        return output;
    }
    
    // Elementgrads returns the gradient of the error with respect to the inputs to this layer.
    // It takes in the gradient of the error WRT the inputs to the next layer (the outputs of this one)
    // and the inputs to this layer.
    
    std::vector<std::vector<double>> elementgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        return std::vector<std::vector<double>>(nextlayergradient);
    }
    
    // Backprop adjusts the weights and other values according to the gradients.
    // It returns the gradient with respect to the inputs of this layer (elementgrads).
    
    std::vector<std::vector<double>> backprop(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs, double alpha = 0.01, bool VERBOSE = false) {
        return nextlayergradient;
    }
    
    // METHODS THAT ARE CONSTANT ACROSS ALL CLASSES
    
    // COMPUTATION METHODS (FORWARD AND BACKWARD PASSING)
    
    std::vector<std::vector<double>> crosscorrelate(std::vector<std::vector<double>> input, std::vector<std::vector<double>> ker) {
        int in = input.size();
        int im = input[0].size();
        int kn = ker.size();
        int km = ker[0].size();
        
        std::vector<std::vector<double>> retval;
        for (int i = 0; i < in - kn + 1; i++) {
            std::vector<double> row;
            for (int j = 0; j < im - km + 1; j++) {
                double res = 0;
                for (int x = 0; x < ker.size(); x++) {
                    for (int y = 0; y < ker[x].size(); y++) res += get(input, x + i, y + j) * ker[x][y];
                }
                row.push_back(res);
            }
            retval.push_back(row);
        }
        return retval;
    }
    
    std::vector<std::vector<double>> convolve(std::vector<std::vector<double>> input, std::vector<std::vector<double>> ker) {
        return crosscorrelate(input, rot(ker, 2));
    }
    // rotates counterclockwise
    std::vector<std::vector<double>> rot(std::vector<std::vector<double>> input, int n) {
        while (n < 0) n += 4;
        n = n % 4;
        if (n == 0) return input;
        auto v = rot(input, n - 1);
        
        std::vector<std::vector<double>> trans(v[0].size(), std::vector<double>(v.size()));
        for (int i = 0; i < v.size(); i++) {
            for (int j = 0; j < v[i].size(); j++) trans[trans.size() - j - 1][i] = v[i][j];
        }
        return trans;
    }
    
    std::vector<std::vector<double>> crosscorrelate_full(std::vector<std::vector<double>> input, std::vector<std::vector<double>> ker) {
        int in = input.size();
        int im = input[0].size();
        int kn = ker.size();
        int km = ker[0].size();
        
        std::vector<std::vector<double>> retval;
        for (int i = 1 - kn; i < in; i++) {
            std::vector<double> row;
            for (int j = 1 - km; j < im; j++) {
                double res = 0;
                for (int x = 0; x < ker.size(); x++) {
                    for (int y = 0; y < ker[x].size(); y++) res += get(input, x + i, y + j) * ker[x][y];
                }
                row.push_back(res);
            }
            retval.push_back(row);
        }
        return retval;
    }
    
    std::vector<std::vector<double>> convolve_full(std::vector<std::vector<double>> input, std::vector<std::vector<double>> ker) {
        return crosscorrelate_full(input, rot(ker, 2));
    }
    
    // AUXILIARY METHODS
    
    double get(std::vector<std::vector<double>> input, int x, int y, bool interp = false) {
        if (interp) {
            int row = std::max(0, std::min((int)(input.size()) - 1, x));
            int col = std::max(0, std::min((int)(input[row].size()) - 1, y));
            return input[row][col];
            
        }
        
        
        if (x < 0 || x >= input.size()) return 0;
        if (y < 0 || y >= input[x].size()) return 0;
        return input[x][y];
    }
    
    
    static std::string vtos(std::vector<std::vector<double>> v) {
        std::string res = "";
        for (auto i : v) {
            res = res + "[ ";
            for (auto j : i) res = res + std::to_string(j) + " ";
            res = res + "]\n";
        }
        return res;
    }
    
    std::string toString() {
        std::string header = "INPUT [" + std::to_string(in_n) + " " + std::to_string(in_m) + "] OUTPUT [" + std::to_string(out_n) + " " + std::to_string(out_m) + "]\n";
        return header;
    }
    
    // Random array
    static std::vector<std::vector<double>> random(int n, int m, double rad) {
        std::vector<std::vector<double>> res(n, std::vector<double>(m));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = rad * (1 - 2 *  (double)(rand()) / (double)(RAND_MAX) );
        }
        return res;
    }
    
    // Random array
    static std::vector<std::vector<double>> randpos(int n, int m, double rad) {
        std::vector<std::vector<double>> res(n, std::vector<double>(m));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = rad * ( (double)(rand()) / (double)(RAND_MAX) );
        }
        return res;
    }
    
    // Constant array
    static std::vector<std::vector<double>> constant(int n, int m, double rad) {
        std::vector<std::vector<double>> res(n, std::vector<double>(m));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = rad;
        }
        return res;
    }
    
    // A - B
    static std::vector<std::vector<double>> diff(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) {
        int n = std::min(a.size(), b.size());
        int m = std::min(a[0].size(), b[0].size());
        std::vector<std::vector<double>> res(n, std::vector<double>(m, 0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res[i][j] = a[i][j] - b[i][j];
        }
        return res;
    }
    
    // Pad a 2 dimdensional array to expand its size using edge elements.
    static std::vector<std::vector<double>> pad(std::vector<std::vector<double>> v, int side) {
        std::vector<std::vector<double>> res(v.size() + side * 2, std::vector<double>(v[0].size() + 2 * side, 0));
        for (int i = 0; i < res.size(); i++) {
            for (int j = 0; j < res[i].size(); j++) {
                int relx = std::max(0, std::min(i - side, (int)(v.size()) - 1));
                int rely = std::max(0, std::min(j - side, (int)(v[relx].size()) - 1));
                res[i][j] = v[relx][rely];
            }
        }
        return res;
    }
};

// Activation LAYERS

class SigmoidLayer : public Layer {
    public:
    SigmoidLayer() {
        out_n = 1;
        out_m = 1;
        in_n = 1;
        in_m = 1;
    }
    
    SigmoidLayer(int a, int b) {
        in_n = a;
        in_m = b;
        out_n = a;
        out_m = b;
    }
    
    SigmoidLayer(const SigmoidLayer& other) {
        out_n = other.out_n;
        out_m = other.out_m;
        in_n = other.in_n;
        in_m = other.in_m;
    }
    
    double activation(double x) {
        return std::tanh(x);
    }
    double deriv(double x, double y) {
        return 1 - (y * y);
    }
    
    std::vector<std::vector<double>> compute(std::vector<std::vector<double>> input) {
        std::vector<std::vector<double>> res;
        for (auto i : input) {
            std::vector<double> v;
            for (auto j : i) v.push_back(activation(j));
            res.push_back(v);
        }
        return res;
    }
    
    std::vector<std::vector<double>> elementgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        std::vector<std::vector<double>> res;
        for (int i = 0; i < nextlayergradient.size() && i < inputs.size(); i++) {
            std::vector<double> v;
            for (int j = 0; j < nextlayergradient[i].size() && j < inputs[i].size(); j++) {
                // (dE/dY) * (dY/dX)
                v.push_back(nextlayergradient[i][j] * deriv(inputs[i][j], activation(inputs[i][j])));
            }
            res.push_back(v);
        }
        return res;
    }
    
    std::vector<std::vector<double>> backprop(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs, double alpha = 0.01, bool VERBOSE = false) {
        return elementgrads(nextlayergradient, inputs);
    }
};

class ReLULayer : public Layer {
    public:
    ReLULayer() {
        out_n = 1;
        out_m = 1;
        in_n = 1;
        in_m = 1;
    }
    
    ReLULayer(int a, int b) {
        in_n = a;
        in_m = b;
        out_n = a;
        out_m = b;
    }
    
    ReLULayer(const ReLULayer& other) {
        out_n = other.out_n;
        out_m = other.out_m;
        in_n = other.in_n;
        in_m = other.in_m;
    }
    
    double activation(double x) {
        return (x > 0) ? x : 0;
    }
    double deriv(double x, double y) {
        return (y == 0) ? 0 : 1;
    }
    
    std::vector<std::vector<double>> compute(std::vector<std::vector<double>> input) {
        std::vector<std::vector<double>> res;
        for (auto i : input) {
            std::vector<double> v;
            for (auto j : i) v.push_back(activation(j));
            res.push_back(v);
        }
        return res;
    }
    
    std::vector<std::vector<double>> elementgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        std::vector<std::vector<double>> res;
        for (int i = 0; i < nextlayergradient.size() && i < inputs.size(); i++) {
            std::vector<double> v;
            for (int j = 0; j < nextlayergradient[i].size() && j < inputs[i].size(); j++) {
                // (dE/dY) * (dY/dX)
                v.push_back(nextlayergradient[i][j] * deriv(inputs[i][j], activation(inputs[i][j])));
            }
            res.push_back(v);
        }
        return res;
    }
    
    std::vector<std::vector<double>> backprop(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs, double alpha = 0.01, bool VERBOSE = false) {
        return elementgrads(nextlayergradient, inputs);
    }
};

#endif

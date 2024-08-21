#ifndef CNN_H
#define CNN_H

#include "MODULAR_LAYER.H"

// Simple implementation of a convolutional neural network. Convolutions do not take elements from outside the input arrays.
// Based on this article https://medium.com/@kattarajesh2001/convolutional-neural-network-from-scratch-0d7513d62923
// This one is more modular -- instead of the entire NN being a class we have classes for layers. You will have to arrange them into the CNN.

// A Convolution Layer uses a kernel of size (n)x(m) to produce the output.
// Given the input array size and the kernel size, the output size is fixed because the kernel moves in a fixed pattern.
class ConvLayer : public Layer {
    public:
    std::vector<std::vector<double>> kernel;
    std::vector<std::vector<double>> bias;
    int n, m;
    
    
    ConvLayer() {
        n = 1;
        m = 1;
        in_n = 1;
        in_m = 1;
        out_n = 1;
        out_m = 1;
        kernel = std::vector<std::vector<double>>(n, std::vector<double>(m, 1));
        bias = std::vector<std::vector<double>>(in_n - n + 1, std::vector<double>(in_m - m + 1, 1));
    }
    
    ConvLayer(int a, int b, int ia, int ib) {
        n = a;
        m = b;
        in_n = ia;
        in_m = ib;
        out_n = in_n - n + 1;
        out_m = in_m - m + 1;
        kernel = std::vector<std::vector<double>>(n, std::vector<double>(m, 1));
        bias = std::vector<std::vector<double>>(out_n, std::vector<double>(out_m, 1));
    }
    
    ConvLayer(std::vector<std::vector<double>> ker, int ia, int ib) {
        n = ker.size();
        m = ker[0].size();
        in_n = ia;
        in_m = ib;
        out_n = in_n - n + 1;
        out_m = in_m - m + 1;
        bias = std::vector<std::vector<double>>(out_n, std::vector<double>(out_m, 1));
        for (auto i : ker) {
            std::vector<double> v;
            for (auto j : i) v.push_back(j);
            kernel.push_back(v);
        }
    }
    
    ConvLayer(std::vector<std::vector<double>> ker, std::vector<std::vector<double>> bia) {
        n = ker.size();
        m = ker[0].size();
        out_n = bia.size();
        out_m = bia[0].size();
        in_n = out_n + n - 1;
        in_m = out_m + n - 1;
        bias = std::vector<std::vector<double>>(out_n, std::vector<double>(out_m, 1));
        for (auto i : ker) {
            std::vector<double> v;
            for (auto j : i) v.push_back(j);
            kernel.push_back(v);
        }
        
        for (int i = 0; i < out_n; i++) {
            for (int j = 0; j < out_m; j++) bias[i][j] = bia[i][j];
        }
    }
    
    ConvLayer(const ConvLayer& other) {
        n = other.kernel.size();
        m = other.kernel[0].size();
        in_n = other.in_n;
        in_m = other.in_m;
        out_n = other.out_n;
        out_m = other.out_m;
        for (auto i : other.kernel) {
            std::vector<double> v;
            for (auto j : i) v.push_back(j);
            kernel.push_back(v);
        }
        
        for (auto i : other.bias) {
            std::vector<double> v;
            for (auto j : i) v.push_back(j);
            bias.push_back(v);
        }
    }
    
    std::vector<std::vector<double>> compute(std::vector<std::vector<double>> input) {
        if (input.size() != in_n) return bias;
        if (input[0].size() != in_m) return bias;
        
        return crosscorrelate(input, kernel);
    }
    
    // BACKPROP
    // https://medium.com/@kattarajesh2001/convolutional-neural-network-from-scratch-0d7513d62923
    // nextlayergradient (dE/dY) is the gradient of the error E with respect to the outputs of this layer Y.
    // Remember that this layer only computes the weighted sum. There are other layers to do other things.
    // The next methods compute gradients with respect to this layer's input X and the kernel K and possibly the bias B
    
    // Y[i][j] = B[i][j] + (SUM over all applicable indices (x, y)) K[x][y] * X[x][y]
    // dE/dK[i][j] = dE/dY * dY/dK = dE/dY * d/dK(KX) = dE/dY * X
    // so this means dE/dK[i][j] = (SUM over all indices (x, y) from previously) dE/dY[rel_x][rel_y] * X[x][y]
    // where (rel_x, rel_y) are the corresponding positions in Y as the window we go for X (e.g. if (x, y) is the top left of the window then (rel_x, rel_y) = (0, 0))
    // But this is actually just correlate(X, dE/dY)
    std::vector<std::vector<double>> kernelgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        return crosscorrelate(inputs, nextlayergradient);
    }
    
    // Next is the gradient with respect to the bias. dE/dB = dE/dY * dY/dB. However since Y[i][j] = B[i][j] + ??? the derivative dY/dB = 1.
    // Thus dE/dB = dE/dY.
    std::vector<std::vector<double>> biasgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        return nextlayergradient;
    }
    
    // And finally the gradient with respect to the inputs. dE/dX = dE/dY * dY/dX
    // Y[i][j] = B[i][j] + (SUM over all applicable indices (x, y)) K[x][y] * X[x][y]
    // dE/dX[i][j] = dE/dY * dY/dX = dE/dY * d/dX(KX) = dE/dY * K
    // For each position (a, b) that the kernel takes in its journey (coord representing the cell in the output) we add dE/dY(a, b) * K(rel_x, rel_y) where (x, y) is a cell in the current Kernel position
    // But that's just convolve(dE/dY, rotate180(kernel))
    std::vector<std::vector<double>> elementgrads(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs) {
        return convolve_full(nextlayergradient, kernel);
    }
    
    std::vector<std::vector<double>> backprop(std::vector<std::vector<double>> nextlayergradient, std::vector<std::vector<double>> inputs, double alpha = 0.01, bool VERBOSE = false) {
        auto kg = kernelgrads(nextlayergradient, inputs);
        auto bg = biasgrads(nextlayergradient, inputs);
        auto eg = elementgrads(nextlayergradient, inputs);
        
        if (VERBOSE) {
            std::cout << "KERNEL GRADS\n" << vtos(kg);
            std::cout << "BIAS GRADS\n" << vtos(bg);
            std::cout << "ELEMENT GRADS\n" << vtos(eg);
            std::cout << "EXISTING KERNEL\n" << vtos(kernel) << "EXISTING BIAS\n" << vtos(bias) << std::endl; 
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) kernel[i][j] -= alpha * kg[i][j];
        }
        
        for (int i = 0; i < out_n; i++) {
            for (int j = 0; j < out_m; j++) bias[i][j] -= alpha * bg[i][j];
        }
        
        return eg;
    }
    
    
    
    std::string toString() {
        std::string header = "KERNEL [" + std::to_string(n) + " " + std::to_string(m) + "] INPUT [" + std::to_string(in_n) + " " + std::to_string(in_m) + "] OUTPUT [" + std::to_string(out_n) + " " + std::to_string(out_m) + "]\n";
        return header + vtos(kernel) + "\nBIAS\n" + vtos(bias) + "\n" + header;
    }
};

// Activation function layers

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



/*

EXAMPLE CODE



#include "cnn.h"

using namespace std;

double randf() {
    return (double)(rand()) / (double)(RAND_MAX);
}

double randrad() {
    return 1.0 - 2.0 * randf();
}

int rad = 16;
int N = 3000;
int MILESTONE = 100;
double LR = 0.01;

void nntest2(bool VERBOSE = false, int threshold = 50 * 16) {
    if (VERBOSE) cout << "BEGIN CNN TEST\n";
    ConvLayer layer(Layer::random(4, 4, 1), Layer::random(7, 7, 1));
    ReLULayer act(7, 7);
    ConvLayer layer2(Layer::random(4, 4, 1), Layer::random(4, 4, 1));
    ReLULayer act2(4, 4);
    ConvLayer layer3(Layer::random(4, 4, 1), Layer::random(1, 1, 1));
    SigmoidLayer act3(1, 1);
    
    int good = 0;
    int bad = 0;
    double diffsum = 0;
    // Test if the four numbers in each quadrant sum up to at least a threshold
    for (int it = 1; it <= N; it++) {
        bool HIT_MS = (it % (N / MILESTONE) == 0) && (it > 0);
        if (VERBOSE && HIT_MS) cout << "+" << it << endl;
        auto input = Layer::random(10, 10, 16);
        vector<vector<double>> desired(1, vector<double>(1, -1));
        double sum = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) sum += input[i][j];
        }
        if (sum < -threshold) desired[0][0] = 1;
        
        auto v = layer.compute(input);
        auto v1 = act.compute(v);
        auto v2 = layer2.compute(v1);
        auto v3 = act2.compute(v2);
        auto v4 = layer3.compute(v3);
        auto v5 = act3.compute(v4);
        
        auto sigrads3 = act3.backprop(Layer::diff(v5, desired), v4);
        auto grads3 = layer3.backprop(sigrads3, v3, LR);
        auto sigrads2 = act2.backprop(grads3, v2);
        auto grads2 = layer2.backprop(sigrads2, v1, LR);
        auto sigrads = act.backprop(grads2, v);
        layer.backprop(sigrads, input, LR);
        
        diffsum += abs(v5[0][0] - desired[0][0]);
        if (abs(v5[0][0] - desired[0][0]) > abs(v5[0][0] + desired[0][0])) bad++;
        else good++;
        
        if (HIT_MS) {
            cout << "GOOD " << good << " | " << "BAD " << bad << "| DIFFSUM " << (diffsum / (N / MILESTONE)) << endl;
            good = 0;
            bad = 0;
            diffsum = 0;
        }
        
        
        if (VERBOSE && HIT_MS) cout << "-" << it << endl;
    }
    
    cout << "FINAL KERNEL AND BIAS\n";
    cout << Layer::vtos(layer.kernel) << "...\n" << Layer::vtos(layer.bias);
}

void nntest(bool VERBOSE = false, int threshold = 81 * 8) {
    if (VERBOSE) cout << "BEGIN CNN TEST\n";
    ConvLayer layer(Layer::random(5, 5, 1), Layer::random(5, 5, 1));
    if (VERBOSE) cout << "BEGIN CNN TEST\n";
    ReLULayer act(5, 5);
    if (VERBOSE) cout << "BEGIN CNN TEST\n";
    ConvLayer layer2(Layer::random(5, 5, 1), Layer::random(1, 1, 1));
    if (VERBOSE) cout << "BEGIN CNN TEST\n";
    SigmoidLayer act2(1, 1);
    if (VERBOSE) cout << "BEGIN CNN TEST\n";
    
    cout << "CONV1\n" << layer.toString() << "CONV2\n" << layer2.toString() << endl;
    
    int good = 0;
    int bad = 0;
    double diffsum = 0;
    // Test if all the numbers sum up to at least a threshold.
    for (int it = 1; it <= N; it++) {
        bool HIT_MS = (it % (N / MILESTONE) == 0) && (it > 0);
        if (VERBOSE && HIT_MS) cout << "+" << it << endl;
        auto input = Layer::random(9, 9, 16);
        vector<vector<double>> desired(1, vector<double>(1, -1));
        double sum = 0;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) sum += input[i][j];
        }
        if (sum > threshold) desired[0][0] = 1;
        
        auto v = layer.compute(input);
        auto vx = act.compute(v);
        auto vxx = layer2.compute(vx);
        auto final = act2.compute(vxx);
        if (VERBOSE && HIT_MS) {
            cout << "INPUT/DESIRED/DIFF\n";
            cout << Layer::vtos(input);
            cout << Layer::vtos(desired);
            cout << Layer::vtos(Layer::diff(vx, desired));
            cout << "V...X\n";
            cout << Layer::vtos(v);
            cout << Layer::vtos(vx);
            cout << Layer::vtos(vxx);
            cout << Layer::vtos(final);
        }
        
        
        auto sigrads2 = act2.backprop(Layer::diff(final, desired), vxx);
        auto grads2 = layer2.backprop(sigrads2, vx, LR);
        auto sigrads = act.backprop(grads2, v);
        layer.backprop(sigrads, input, LR);
        
        diffsum += abs(final[0][0] - desired[0][0]);
        if (abs(final[0][0] - desired[0][0]) > abs(final[0][0] + desired[0][0])) bad++;
        else good++;
        
        if (HIT_MS) {
            cout << "GOOD " << good << " | " << "BAD " << bad << "| DIFFSUM " << (diffsum / (N / MILESTONE)) << endl;
            good = 0;
            bad = 0;
            diffsum = 0;
        }
        
        
        if (VERBOSE && HIT_MS) cout << "-" << it << endl;
    }
    
    cout << "FINAL KERNEL AND BIAS\n";
    cout << Layer::vtos(layer.kernel) << "...\n" << Layer::vtos(layer.bias);
}

void layertest(bool VERBOSE = false) {
    if (VERBOSE) cout << "BEGIN LAYER TEST\n";
    ConvLayer layer(Layer::random(2, 2, 16), Layer::random(2, 2, 16));
    cout << layer.toString();
    
    double diffsum = 0;
    // Test if the four numbers in each quadrant sum up to at least a threshold
    for (int it = 1; it <= N; it++) {
        bool HIT_MS = (it % (N / MILESTONE) == 0) && (it > 0);
        if (VERBOSE && HIT_MS) cout << "+" << it << endl;
        auto input = Layer::random(3, 3, 1);
        vector<vector<double>> desired(2, vector<double>(2, -1));
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                desired[i][j] = (input[i][j] + input[i + 1][j + 1] + input[i][j + 1] + input[i + 1][j]);
            }
        }
        
        auto v = layer.compute(input);
        
        if (VERBOSE && HIT_MS) {
            cout << "INPUT/V/DESIRED/DIFF\n";
            cout << Layer::vtos(input);
            cout << Layer::vtos(v);
            cout << Layer::vtos(desired);
            cout << Layer::vtos(Layer::diff(v, desired));
        }
        
        layer.backprop(Layer::diff(v, desired), input, 0.01, VERBOSE && HIT_MS);
        
        for (int i = 0; i < 4; i++) {
            diffsum += abs(desired[i / 2][i % 2] - v[i / 2][i % 2]);
            if (VERBOSE && HIT_MS) cout << desired[i / 2][i % 2] << " " <<  v[i / 2][i % 2] << endl;
        }
        
        if (HIT_MS) {
            cout << "DIFFSUM " << diffsum << "\n";
            diffsum = 0;
        }
        
        
        if (VERBOSE && HIT_MS) cout << "-" << it << endl;
    }
    
    cout << "FINAL KERNEL AND BIAS\n";
    cout << Layer::vtos(layer.kernel) << "...\n" << Layer::vtos(layer.bias);
}

int main()
{
    srand(time(0));
    
    vector<vector<double>> v({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    vector<vector<double>> i({{1, 2}, {3, 4}});
    vector<vector<double>> s({{100}});
    
    // v = Layer::random(3, 3, 16);
    // i = Layer::random(2, 2, 16);
    cout << Layer::vtos(v) << Layer::vtos(i) << " " << Layer::vtos(s);
    Layer l;
    cout << Layer::vtos(l.convolve_full(i, s));
    
    
    
    nntest2(false);
    
    return 0;
}

*/

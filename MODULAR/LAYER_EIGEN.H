#ifndef LAYER_EIGEN_H
#define LAYER_EIGEN_H

#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <ctime>
#include <cmath>

#include <Eigen/Dense>

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
    
    Eigen::MatrixXd compute(Eigen::MatrixXd input) {
        Eigen::MatrixXd output(out_n, out_m);
        for (int i = 0; i < in_n && i < out_n; i++) {
            for (int j = 0; j < in_m && j < out_m; j++) output(i, j) = input(i, j);
        }
        return output;
    }
    
    // Elementgrads returns the gradient of the error with respect to the inputs to this layer.
    // It takes in the gradient of the error WRT the inputs to the next layer (the outputs of this one)
    // and the inputs to this layer.
    
    Eigen::MatrixXd elementgrads(Eigen::MatrixXd nextlayergradient, Eigen::MatrixXd inputs) {
        return Eigen::MatrixXd(nextlayergradient);
    }
    
    // Backprop adjusts the weights and other values according to the gradients.
    // It returns the gradient with respect to the inputs of this layer (elementgrads).
    
    Eigen::MatrixXd backprop(Eigen::MatrixXd nextlayergradient, Eigen::MatrixXd inputs, double alpha = 0.01, bool VERBOSE = false) {
        return nextlayergradient;
    }
    
    // METHODS THAT ARE CONSTANT ACROSS ALL CLASSES
    
    // COMPUTATION METHODS (FORWARD AND BACKWARD PASSING)
    
    Eigen::MatrixXd crosscorrelate(Eigen::MatrixXd input, Eigen::MatrixXd ker) {
        int in = input.rows();
        int im = input.cols();
        int kn = ker.rows();
        int km = ker.cols();
        
        Eigen::MatrixXd retval(in - kn + 1, im - km + 1);
        for (int i = 0; i < in - kn + 1; i++) {
            for (int j = 0; j < im - km + 1; j++) {
                double res = 0;
                for (int x = 0; x < ker.rows(); x++) {
                    for (int y = 0; y < ker.cols(); y++) res += get(input, x + i, y + j) * ker(x, y);
                }
                retval(i, j) = res;
            }
        }
        return retval;
    }
    
    Eigen::MatrixXd convolve(Eigen::MatrixXd input, Eigen::MatrixXd ker) {
        return crosscorrelate(input, rot(ker, 2));
    }
    // rotates counterclockwise
    Eigen::MatrixXd rot(Eigen::MatrixXd input, int n) {
        while (n < 0) n += 4;
        n = n % 4;
        if (n == 0) return input;
        auto v = rot(input, n - 1);
        
        Eigen::MatrixXd trans(v.cols(), v.rows());
        for (int i = 0; i < v.rows(); i++) {
            for (int j = 0; j < v.cols(); j++) trans(trans.rows() - j - 1, i) = v(i, j);
        }
        return trans;
    }
    
    Eigen::MatrixXd crosscorrelate_full(Eigen::MatrixXd input, Eigen::MatrixXd ker) {
        int in = input.rows();
        int im = input.cols();
        int kn = ker.rows();
        int km = ker.cols();
        
        Eigen::MatrixXd retval(in + kn - 1, im + km - 1);
        for (int i = 1 - kn; i < in; i++) {
            for (int j = 1 - km; j < im; j++) {
                double res = 0;
                for (int x = 0; x < ker.rows(); x++) {
                    for (int y = 0; y < ker.cols(); y++) res += get(input, x + i, y + j) * ker(x, y);
                }
                retval(i + kn - 1, j + km - 1) = res;
            }
        }
        return retval;
    }
    
    Eigen::MatrixXd convolve_full(Eigen::MatrixXd input, Eigen::MatrixXd ker) {
        return crosscorrelate_full(input, rot(ker, 2));
    }
    
    // AUXILIARY METHODS
    
    double get(Eigen::MatrixXd input, int x, int y, bool interp = false) {
        if (interp) {
            int row = std::max(0, std::min((int)(input.rows()) - 1, x));
            int col = std::max(0, std::min((int)(input.cols()) - 1, y));
            return input(row, col);
            
        }
        
        
        if (x < 0 || x >= input.rows()) return 0;
        if (y < 0 || y >= input.cols()) return 0;
        return input(x, y);
    }
    
    
    static std::string vtos(Eigen::MatrixXd v) {
        std::string res = "";
        for (int i = 0; i < v.rows(); i++) {
            res = res + "[ ";
            for (int j = 0; j < v.cols(); j++) res = res + std::to_string(v(i, j)) + " ";
            res = res + "]\n";
        }
        return res;
    }
    
    std::string toString() {
        std::string header = "INPUT [" + std::to_string(in_n) + " " + std::to_string(in_m) + "] OUTPUT [" + std::to_string(out_n) + " " + std::to_string(out_m) + "]\n";
        return header;
    }
    
    // Random array
    static Eigen::MatrixXd random(int n, int m, double rad) {
        Eigen::MatrixXd res(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res(i, j) = rad * (1 - 2 *  (double)(rand()) / (double)(RAND_MAX) );
        }
        return res;
    }
    
    // Random array
    static Eigen::MatrixXd randpos(int n, int m, double rad) {
        Eigen::MatrixXd res(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res(i, j) = rad * ( (double)(rand()) / (double)(RAND_MAX) );
        }
        return res;
    }
    
    // Constant array
    static Eigen::MatrixXd constant(int n, int m, double rad) {
        return Eigen::MatrixXd::Constant(n, m, rad);
    }
    
    // A - B
    static Eigen::MatrixXd diff(Eigen::MatrixXd a, Eigen::MatrixXd b) {
        int n = std::min(a.rows(), b.rows());
        int m = std::min(a.cols(), b.cols());
        Eigen::MatrixXd res(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res(i, j) = a(i, j) - b(i, j);
        }
        return res;
    }
    
    // Pad a 2 dimdensional array to expand its size using edge elements.
    static Eigen::MatrixXd pad(Eigen::MatrixXd v, int side) {
        Eigen::MatrixXd res(v.rows() + side * 2, v.cols() + 2 * side);
        for (int i = 0; i < res.rows(); i++) {
            for (int j = 0; j < res.cols(); j++) {
                int relx = std::max(0, std::min(i - side, (int)(v.rows()) - 1));
                int rely = std::max(0, std::min(j - side, (int)(v.cols()) - 1));
                res(i, j) = v(relx, rely);
            }
        }
        return res;
    }

    static Eigen::MatrixXd padconst(Eigen::MatrixXd v, int side, double val) {
        Eigen::MatrixXd res(v.rows() + side * 2, v.cols() + 2 * side);
        for (int i = 0; i < res.rows(); i++) {
            for (int j = 0; j < res.cols(); j++) {
                int relx = i - side;
                int rely = j - side;
                if (relx < 0 || rely < 0 || relx >= v.rows() || rely >= v.cols()) res(i, j) = val;
                else res(i, j) = v(relx, rely);
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
    
    SigmoidLayer(int a) {
        in_n = a;
        in_m = 1;
        out_n = a;
        out_m = 1;
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
    
    Eigen::MatrixXd compute(Eigen::MatrixXd input) {
        Eigen::MatrixXd res(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) res(i, j) = (activation(input(i, j)));
        }
        return res;
    }
    
    Eigen::MatrixXd elementgrads(Eigen::MatrixXd nextlayergradient, Eigen::MatrixXd inputs) {
        Eigen::MatrixXd res(nextlayergradient.rows(), nextlayergradient.cols());
        for (int i = 0; i < nextlayergradient.rows() && i < inputs.rows(); i++) {
            for (int j = 0; j < nextlayergradient.cols() && j < inputs.cols(); j++) {
                // (dE/dY) * (dY/dX)
                res(i, j) = nextlayergradient(i, j) * deriv(inputs(i, j), activation(inputs(i, j)));
            }
        }
        return res;
    }
    
    Eigen::MatrixXd backprop(Eigen::MatrixXd nextlayergradient, Eigen::MatrixXd inputs, double alpha = 0.01, bool VERBOSE = false) {
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
    
    ReLULayer(int a) {
        in_n = a;
        in_m = 1;
        out_n = a;
        out_m = 1;
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
    
    Eigen::MatrixXd compute(Eigen::MatrixXd input) {
        Eigen::MatrixXd res(input.rows(), input.cols());
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < input.cols(); j++) res(i, j) = (activation(j));
        }
        return res;
    }
    
    Eigen::MatrixXd elementgrads(Eigen::MatrixXd nextlayergradient, Eigen::MatrixXd inputs) {
        Eigen::MatrixXd res(nextlayergradient.rows(), nextlayergradient.cols());
        for (int i = 0; i < nextlayergradient.rows() && i < inputs.rows(); i++) {
            for (int j = 0; j < nextlayergradient.cols() && j < inputs.cols(); j++) {
                // (dE/dY) * (dY/dX)
                res(i, j) = nextlayergradient(i, j) * deriv(inputs(i, j), activation(inputs(i, j)));
            }
        }
        return res;
    }
    
    Eigen::MatrixXd backprop(Eigen::MatrixXd nextlayergradient, Eigen::MatrixXd inputs, double alpha = 0.01, bool VERBOSE = false) {
        return elementgrads(nextlayergradient, inputs);
    }
};

#endif

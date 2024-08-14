#ifndef NEURAL_EIGEN_H
#define NEURAL_EIGEN_H

#include <iostream>
#include <Eigen/Dense>
 
#include <vector>
#include <cfloat>
#include <climits>
#include <algorithm>
#include <cmath>
// Implementation of a small evolving neural network system WITH A SINGLE OUTPUT

#define DEFAULT_INPUT 2
#define DEFAULT_LAYERS 2
#define DEFAULT_HIDDEN 2
#define INF (100000000)

class NeuralNetwork {
    public:
    
    // The mechanism of a neural network is actually fairly simple. 
    // There are input nodes, hidden nodes, and output nodes. 
    // Hidden and output nodes simply receive data in the form of a linear combination of the data from their parents.
    // The node then puts this linear combination (input data) into an activation function to "accentuate" the value.
    // Nodes then send the activated data to the child nodes as part of the linear combinations of those child nodes.
    // Each hidden layer and output layer node also takes in a bias coefficient. The bias coefficient is represented as a node in each layer that always outputs one.
    
    // The output layer has only one node.
    // The activation function for each layer is a sigmoid.
    
    // How coefficients are encoded? weights[L][a][b] is the scale the data from node a in layer L 
    // is multiplied when inserted into node b in layer L + 1.
    
    // This implementation has 2 activation functions: one for the output layer and one for everything else.
    
    int INPUT_SIZE = DEFAULT_INPUT;
    int HIDDEN_LAYERS = DEFAULT_LAYERS;
    int NODES_PER_HIDDEN = DEFAULT_HIDDEN;
    
    double WEIGHTLIMIT = (1<<16);
    
    int edges = 0;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> values;

    void init() {
        edges = 0;
        for (auto i : weights) edges += i.rows() * i.cols();
        
        values = std::vector<Eigen::VectorXd>();
        values.push_back(Eigen::VectorXd(INPUT_SIZE + 1, 1));
        for (int i = 0; i < HIDDEN_LAYERS; i++) values.push_back(Eigen::VectorXd(NODES_PER_HIDDEN + 1, 1));
        values.push_back(Eigen::VectorXd(1, 1));
        
        for (int i = 0; i <= HIDDEN_LAYERS; i++) values[i](values[i].cols() - 1) = 1;
    }
    
    NeuralNetwork(const NeuralNetwork& other) {
        INPUT_SIZE = other.INPUT_SIZE;
        HIDDEN_LAYERS = other.HIDDEN_LAYERS;
        NODES_PER_HIDDEN = other.NODES_PER_HIDDEN;
        weights = std::vector<Eigen::MatrixXd>();
        for (int i = 0; i < other.weights.size(); i++) {
            weights.push_back(Eigen::MatrixXd(other.weights[i]));
        }
        
        init();
    }

    NeuralNetwork() {
        if (HIDDEN_LAYERS == 0) {
            weights.push_back(Eigen::MatrixXd::Constant(INPUT_SIZE + 1, 1, 1));
            return;
        }
        weights = std::vector<Eigen::MatrixXd>(1, Eigen::MatrixXd::Constant(INPUT_SIZE + 1, NODES_PER_HIDDEN, 1));
        for (int i = 1; i < HIDDEN_LAYERS; i++) {
            weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, NODES_PER_HIDDEN, 1));
        }
        weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, 1, 1));
        
        init();
    }
    
    NeuralNetwork(int protogen, int primagen, int primogenitor) {
        INPUT_SIZE = protogen;
        HIDDEN_LAYERS = primagen;
        NODES_PER_HIDDEN = primogenitor;
        
        if (HIDDEN_LAYERS == 0) {
            weights.push_back(Eigen::MatrixXd::Constant(INPUT_SIZE + 1, 1, 1));
            return;
        }
        weights = std::vector<Eigen::MatrixXd>(1, Eigen::MatrixXd::Constant(INPUT_SIZE + 1, NODES_PER_HIDDEN, 1));
        for (int i = 1; i < HIDDEN_LAYERS; i++) {
            weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, NODES_PER_HIDDEN, 1));
        }
        weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, 1, 1));
        
        init();
    }
    
    double sigmoid(double x) {
        return std::tanh(x);
    }
    
    double sigd(double y) {
        return 1 - (y * y);
    }
    
    double activation(double x) {
        // return x;
        return sigmoid(x);
    }
    
    
    double activd(double y) {
        // return 1;
        return sigd(y);
    }
    
    double finalactivation(double x) {
        return sigmoid(x);
        return x;
    }
    
    double finalad(double y) {
        return sigd(y);
        return 1;
    }
    
    double eval(std::vector<double> input, bool VERBOSE = false) {
        if (input.size() < INPUT_SIZE) return DBL_MIN;
        if (HIDDEN_LAYERS == 0) {
            double res = 0;
            for (int i = 0; i < INPUT_SIZE; i++) res += input[i] * weights[0](i, 0);
            res += weights[0](INPUT_SIZE, 0);
            return activation(res);
        }

        Eigen::VectorXd data(INPUT_SIZE);
        
        values[0] = Eigen::VectorXd(INPUT_SIZE + 1);
        values[0](INPUT_SIZE) = 1;

        if (VERBOSE) {
            std::cout << "INPUT VALUES\n";
            std::cout << values[0] << "\n";
        }
        
        
        for (int i = 0; i < INPUT_SIZE; i++) {
            values[0](i) = input[i];
            data(i) = input[i];
        }

        if (VERBOSE) {
            std::cout << "DATA VEC\n";
            std::cout << data << "\n";
        }
        
        Eigen::VectorXd indata(INPUT_SIZE + 1);
        
        indata << data, Eigen::VectorXd(1);
        indata(INPUT_SIZE) = 1;

        if (VERBOSE) std::cout << "INDATA\n" << indata << "\n";

        data = indata.transpose() * weights[0];

        for (int i = 0; i < data.rows(); i++) data(i) = activation(data(i));

        values[1] = Eigen::VectorXd(NODES_PER_HIDDEN + 1);
        values[1] << data, Eigen::VectorXd(1, 1);
        values[1](NODES_PER_HIDDEN) = 1;
        if (VERBOSE) std::cout << "DATA\n" << data << "\n";
        
        Eigen::VectorXd newdata(NODES_PER_HIDDEN + 1);
        
        for (int layer = 1; layer < HIDDEN_LAYERS; layer++) {
            newdata = Eigen::VectorXd(NODES_PER_HIDDEN + 1);
            newdata << data, Eigen::VectorXd(1, 1);
            newdata(NODES_PER_HIDDEN) = 1;


            newdata = newdata.transpose() * weights[layer];

            for (int i = 0; i < newdata.rows(); i++) newdata(i) = activation(newdata(i));
            
            data = newdata;
            values[layer + 1] = Eigen::VectorXd(NODES_PER_HIDDEN + 1);
            values[layer + 1] << data, Eigen::VectorXd(1, 1);
            values[layer + 1](NODES_PER_HIDDEN) = 1;


            if (VERBOSE) std::cout << "NEWDATA\n" << newdata << "\n";
            // for (auto i : data) std::cout << i << " ";
            // std::cout << "\n";
        }

        Eigen::VectorXd data2 = Eigen::VectorXd(NODES_PER_HIDDEN + 1);
        data2 << data, Eigen::VectorXd(1, 1);
        data2(NODES_PER_HIDDEN) = 1;

        data = data2;

        if (VERBOSE) {

            std::cout << "FINAL LAYER\n";

            std::cout << data.rows() << " " << data.cols() << "\n";
            std::cout << weights[HIDDEN_LAYERS].rows() << " " << weights[HIDDEN_LAYERS].cols() << "\n";
            std::cout << data.transpose() << "\n" << weights[HIDDEN_LAYERS].transpose() << "\n";
        }
        
        double res = (data.transpose() * weights[HIDDEN_LAYERS])(0, 0);
        if (VERBOSE) std::cout << res << std::endl;
        
        values[HIDDEN_LAYERS + 1](0) = finalactivation(res); // store the final value for consistency

        if (VERBOSE) {
            std::cout << "VALUES\n";
            for (auto v : values) std::cout << ">" << v << "\n";
        }
                
        return finalactivation(res);
    }

    std::string toString() {
        std::string res = "[" + std::to_string(INPUT_SIZE) + " " + std::to_string(HIDDEN_LAYERS) + " " + std::to_string(NODES_PER_HIDDEN) + "]\n";
        
        for (int i = 0; i < weights.size(); i++) {
            res = res + "\nLAYER " + std::to_string(i) + ":\n";
            for (int j = 0; j < weights[i].rows(); j++) {
                for (int k = 0; k < weights[i].cols(); k++) res = res + "" + std::to_string(weights[i](j, k)) + " ";
                res = res + "\n";
            }
        }
        
        return res;
    }

    void backpropsimple(double yhat, double y, double alpha, bool verbose = false) {
        if (verbose) {
            std::cout << "NN\n";
            std::cout << toString() << "\n";
            std::cout << "NN VALUES\n";
            for (auto i : values) std::cout << i << "\n";
        }
        
        
        double Eprime = yhat - y; // d(Squared error) / d(yhat) = dE / dY'

        if (verbose) std::cout << "HERE ARE THE THINGS " << yhat << " " << y << " " << Eprime << "\n";
        
        // denote N as the input value to a node (weighted sum) and N' the corresponding output (activation(N))
        
        // If a node N outputs to some outputs [x1 ... xk] and inputs from some inputs [z1 ... zj]
        // then the gradient d(squared error) / dN' is simply the sum of the following:
        // d(squared error) / d(x') * dx' / dx * dx / dN' over all x in [x1 ... xk]
        // You can get the values of N' from the values vector which stores all input and intermediate values
        
        // The neural network has N + 2 layers. 1 input layer, N hidden layers, and 1 output layer.
        // There are N + 1 layers of weights. Layer i (weights[i]) forms a matrix of weights from Layer i to Layer i + 1
        // weights[i][j][k] is the weight connecting node j in layer i to node k in layer i + 1
        
        std::vector<std::vector<double> > nodegrads(1, std::vector<double>(INPUT_SIZE, 0)); // Gradients d(squared error) / dN'
        for (int i = 0; i < HIDDEN_LAYERS; i++) nodegrads.push_back(std::vector<double>(NODES_PER_HIDDEN, 0));
        nodegrads.push_back(std::vector<double>(1, 0));
        
        // nodegrads[i][j] is d(squared error) / d(value' of node j in layer i so after the sigmoid)
        
        NeuralNetwork gradients(*this); // weight gradients. 
        // gradients.weights[l][i][j] is the gradient of the edge starting on node i on layer l leading into node j on layer l + 1
        
        // For the output layer it simply has d(sqerror) / d(output')
        nodegrads[HIDDEN_LAYERS + 1][0] = Eprime;
        // Now for successive layers:
        for (int i = HIDDEN_LAYERS; i >= 0; i--) {
            if (verbose) {
                std::cout << "LAYER " << i + 1 << " TO " << i << "\n";
                std::cout << "SHAPE OF LAYER " << weights[i].rows() << " " << weights[i].cols() << "\n";
                std::cout << "SHAPE OF OUTPUT VALUES " << values[i + 1].rows() << "\n";
                std::cout << "SHAPE OF INPUT VALUES " << values[i].rows() << "\n";
                std::cout << "NUMBER OF GRADIENTS TO COMPUTE (VALUE OF K) " << values[i].rows() - 1 << "\n";
                std::cout << "NUMBER OF COMPONENTS IN GRADIENT (VALUE OF J) " << nodegrads[i + 1].size() << "\n";
                std::cout << "VALUES TO CONSIDER " << values[i + 1].transpose() << "\n";
                std::cout << "WEIGHTS TO CONSIDER " << weights[i] << "\n";
            }
            for (int k = 0; k < values[i].rows() - 1; k++) { // node k on layer i
                double sum = 0;
                for (int j = 0; j < nodegrads[i + 1].size(); j++) {
                    if (i == HIDDEN_LAYERS) sum += nodegrads[i + 1][j] * finalad(values[i + 1](j)) * weights[i](k, j);
                    else sum += nodegrads[i + 1][j] * activd(values[i + 1](j)) * weights[i](k, j);
                    if (verbose) {
                        std::cout << "!" << nodegrads[i + 1][j] << " " << values[i + 1](j) << " >> " << (i == HIDDEN_LAYERS ? finalad(values[i + 1](j)) : activd(values[i + 1](j))) << " " << weights[i](k, j) << "\n";
                    }
                }
                nodegrads[i][k] = sum;
            }
        }

        if (verbose) {
            std::cout << "NODE GRADIENTS\n";
            for (auto i : nodegrads) {
                for (auto j : i) std::cout << j << " ";
                std::cout << "\n";
            }
        }
        
        // Now to look at the weights
        
        for (int i = HIDDEN_LAYERS; i >= 0; i--) { // layer i --> layer (i + 1)
            for (int j = 0; j < weights[i].rows(); j++) {
                for (int k = 0; k < weights[i].cols(); k++) {
                    // d(sqerror) / d(weight[i][j][k]) = d(sqerror) / d(v'[i + 1][k]) * d(v'[i + 1][k]) / d(v[i + 1][k]) * d(v[i + 1][k]) / d(w)
                    if (i == HIDDEN_LAYERS) gradients.weights[i](j, k) = nodegrads[i + 1][k] * finalad(values[i + 1](k)) * values[i](j);
                    else gradients.weights[i](j, k) = nodegrads[i + 1][k] * activd(values[i + 1](k)) * values[i](j);
                }
            }
        }
        
        if (verbose) std::cout << "GRADIENTS " << "\n" << gradients.toString() << "\n";
        
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].rows(); j++) {
                for (int k = 0; k < weights[i].cols(); k++) {
                    weights[i](j, k) -= gradients.weights[i](j, k) * alpha;
                    if (weights[i](j, k) < -1 * WEIGHTLIMIT) weights[i](j, k) = -1 * WEIGHTLIMIT;
                    if (weights[i](j, k) > WEIGHTLIMIT) weights[i](j, k) = WEIGHTLIMIT;
                }
            }
        }
    }
    
    private:

    static int find(std::string value, char c, int start = 0) {
        for (int i = start; i < value.length(); i++) {
            if (value[i] == c) return i;
        }
        return value.length();
    }

    static std::string substring(std::string data, int a, int b) { // [a, b)
        return data.substr(a, b - a);
    }
    
    // Generates a neural network based on the toString readout of another.
    
    public:
    
    static NeuralNetwork readIn(std::string data) {
        int space = find(data, ' ');
        int input = std::stoi(substring(data, find(data, '[') + 1, space));
        int space2 = find(data, ' ', space + 1);
        int layers = std::stoi(substring(data, space + 1, space2));
        int hidden = std::stoi(substring(data, space2 + 1, find(data, ']')));
    
        NeuralNetwork nn(input, layers, hidden);
    
        int start = find(data, ':') + 1;
        int previouslayer = start;
        for (int in = 0; in <= input; in++) {
            for (int out = 0; out < hidden; out++) {
                int end = find(data, ' ', start);
                nn.weights[0](in, out) = std::stod(substring(data, start, end));
                start = end + 1;
            }
        }
    
        for (int layer = 1; layer < layers; layer++) {
            start = find(data, ':', previouslayer) + 1;
            previouslayer = start;
            for (int in = 0; in <= hidden; in++) {
                for (int out = 0; out < hidden; out++) {
                    int end = find(data, ' ', start);
                    nn.weights[layer](in, out) = std::stod(substring(data, start, end));
                    start = end + 1;
                }
            }
        }
    
    
        start = find(data, ':', previouslayer) + 1;
        for (int in = 0; in <= hidden; in++) {
            int end = find(data, ' ', start);
            nn.weights[layers](in, 0) = std::stod(substring(data, start, end));
            start = end + 1;
        }
    
        return nn;
    }
};

namespace Genetic {

double randf() {
    return (double)(rand()) / (double)(RAND_MAX);
}

double randrad() {
    return 2 * (randf() - 0.5);
}

NeuralNetwork randomAI(double radius = 1, int protogen = DEFAULT_INPUT, int primagen = DEFAULT_LAYERS, int primogenitor = DEFAULT_HIDDEN) {
    NeuralNetwork nn(protogen, primagen, primogenitor);
    for (int i = 0; i < nn.weights.size(); i++) {
        for (int j = 0; j < nn.weights[i].rows(); j++) {
            for (int k = 0; k < nn.weights[i].cols(); k++) nn.weights[i](j, k) = radius * randrad();
        }
    }
    return nn;
}

NeuralNetwork cross(NeuralNetwork n1, NeuralNetwork n2) {
    NeuralNetwork res(n1);
    for (int i = 0; i < n1.weights.size(); i++) {
        for (int j = 0; j < n1.weights[i].rows(); j++) {
            for (int k = 0; k < n1.weights[i].cols(); k++) if (rand() % 2 == 0) res.weights[i](j, k) = n2.weights[i](j, k);
        }
    }
    return res;
}

NeuralNetwork mutate(NeuralNetwork nn, double radius = 64) {
    int threshold = (int)(nn.edges);
    
    NeuralNetwork res(nn);
    int beep = rand() % threshold;
    int count = 0;
    for (int i = 0; i < nn.weights.size(); i++) {
        for (int j = 0; j < nn.weights[i].rows(); j++) {
            for (int k = 0; k < nn.weights[i].cols(); k++) {
                if (rand() % threshold == 0) res.weights[i](j, k) = radius * randrad();
				if (count == beep) res.weights[i](j, k) = radius * randrad();
				count++;
			}
        }
    }
    return res;
}

}

#endif

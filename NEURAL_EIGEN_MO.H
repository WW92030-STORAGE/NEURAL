#ifndef NEURAL_MO_H
#define NEURAL_MO_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cfloat>
#include <climits>
#include <algorithm>
#include <cmath>
// Implementation of a small evolving neural network system WITH MULTIPLE OUTPUTS

#define DEFAULT_INPUT 2
#define DEFAULT_LAYERS 1
#define DEFAULT_HIDDEN 2
#define DEFAULT_OUTPUT 2
#define INF (100000000)
#define REFRESH_RATE (1.0 / 60.0)

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
    int OUTPUT_SIZE = DEFAULT_OUTPUT;
    
    double WEIGHTLIMIT = (1<<16);
    
    int edges = 0;
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> values;

    void init() {
        edges = 0;
        for (auto i : weights) {
            edges += i.rows() * i.cols();
        }
        
        values = std::vector<Eigen::VectorXd>();
        values.push_back(Eigen::VectorXd(INPUT_SIZE + 1));
        for (int i = 0; i < HIDDEN_LAYERS; i++) values.push_back(Eigen::VectorXd(NODES_PER_HIDDEN + 1));
        values.push_back(Eigen::VectorXd(OUTPUT_SIZE));
        
        for (int i = 0; i <= HIDDEN_LAYERS; i++) values[i](values[i].size() - 1) = 1;
    }
    
    NeuralNetwork(const NeuralNetwork& other) {
        INPUT_SIZE = other.INPUT_SIZE;
        HIDDEN_LAYERS = other.HIDDEN_LAYERS;
        NODES_PER_HIDDEN = other.NODES_PER_HIDDEN;
        OUTPUT_SIZE = other.OUTPUT_SIZE;
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
        weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, OUTPUT_SIZE, 1));
        
        init();
    }
    
    NeuralNetwork(int protogen, int primagen, int primogenitor, int zenith) {
        INPUT_SIZE = protogen;
        HIDDEN_LAYERS = primagen;
        NODES_PER_HIDDEN = primogenitor;
        OUTPUT_SIZE = zenith;
        
        if (HIDDEN_LAYERS == 0) {
            weights.push_back(Eigen::MatrixXd::Constant(INPUT_SIZE + 1, 1, 1));
            return;
        }
        weights = std::vector<Eigen::MatrixXd>(1, Eigen::MatrixXd::Constant(INPUT_SIZE + 1, NODES_PER_HIDDEN, 1));
        for (int i = 1; i < HIDDEN_LAYERS; i++) {
            weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, NODES_PER_HIDDEN, 1));
        }
        weights.push_back(Eigen::MatrixXd::Constant(NODES_PER_HIDDEN + 1, OUTPUT_SIZE, 1));
        
        init();
    }
    
    double sigmoid(double x) {
        return std::tanh(x);
    }
    
    double sigd(double y) {
        return 1 - y * y;
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
    
    std::vector<double> eval(std::vector<double> input, bool VERBOSE = false) {
        if (input.size() < INPUT_SIZE) return std::vector<double>(OUTPUT_SIZE, -1 * DBL_MAX);
        if (HIDDEN_LAYERS == 0) {
			std::vector<double> res(OUTPUT_SIZE, 0);
			for (int ii = 0; ii < OUTPUT_SIZE; ii++) {
            for (int i = 0; i < INPUT_SIZE; i++) res[ii] += input[i] * weights[0](i, ii);
            res[ii] += weights[0](INPUT_SIZE, ii);
			res[ii] = activation(res[ii]);
			}
            return res;

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
        
        Eigen::VectorXd res = (data.transpose() * weights[HIDDEN_LAYERS]);
        if (VERBOSE) std::cout << "FINAL SUMS " << res << std::endl;

		for (int i = 0; i < res.rows(); i++) res(i) = finalactivation(res(i));

        if (VERBOSE) std::cout << "EXPORTING..." << res << " = " << values.size() - HIDDEN_LAYERS <<  "\n";
        
        values[HIDDEN_LAYERS + 1] = Eigen::VectorXd(res); // store the final value for consistency

        if (VERBOSE) std::cout << "EXPORTING...\n";

        std::vector<double> retval;
        for (int i = 0; i < res.rows(); i++) retval.push_back(res(i));

        if (VERBOSE) {
            std::cout << "VALUES\n";
            for (auto v : values) std::cout << ">" << v << "\n";
        }
                
        return retval;
    }
    
    std::string toString() {
        std::string res = "[" + std::to_string(INPUT_SIZE) + " " + std::to_string(HIDDEN_LAYERS) + " ";
        res = res + std::to_string(NODES_PER_HIDDEN) + "] " + std::to_string(OUTPUT_SIZE) + "\n";
        
        for (int i = 0; i < weights.size(); i++) {
            res = res + "\nLAYER " + std::to_string(i) + ":\n";
            for (int j = 0; j < weights[i].rows(); j++) {
                for (int k = 0; k < weights[i].cols(); k++) res = res + "" + std::to_string(weights[i](j, k)) + " ";
                res = res + "\n";
            }
        }
        
        return res;
    }
    
    std::string shape() {
        std::string res = "[" + std::to_string(INPUT_SIZE) + " " + std::to_string(HIDDEN_LAYERS) + " ";
        res = res + std::to_string(NODES_PER_HIDDEN) + "] " + std::to_string(OUTPUT_SIZE);
        return res;
    }
    
    void backpropsimple(std::vector<double> yhat, std::vector<double> y, double alpha, bool verbose = false) {
        if (verbose) {
            std::cout << "NN\n";
            std::cout << toString() << "\n";
            std::cout << "NN VALUES\n";
            for (auto i : values) {
                std::cout << ">" << i << "\n";
            }
        }
        
        
        std::vector<double> Eprime(OUTPUT_SIZE); // d(Squared error) / d(yhat) = dE / dY'
        for (int i = 0; i < yhat.size() && i < y.size(); i++) Eprime[i] = (yhat[i] - y[i]); // The partial derivative, only the parts that contain what we are differentiating against matter.
        
        // denote N as the input value to a node (weighted sum) and N' the corresponding output (activation(N))
        
        // If a node N outputs to some outputs [x1 ... xk] and inputs from some inputs [z1 ... zj]
        // then the gradient d(squared error) / dN' is simply the sum of the following:
        // d(squared error) / d(z') * dz' / dz * dz / dN' over all z in [z1 ... zj]
        // You can get the values of N' from the values vector which stores all input and intermediate values
        
        // The neural network has N + 2 layers. 1 input layer, N hidden layers, and 1 output layer.
        // There are N + 1 layers of weights. Layer i (weights[i]) forms a matrix of weights from Layer i to Layer i + 1
        // weights[i][j][k] is the weight connecting node j in layer i to node k in layer i + 1
        
        std::vector<std::vector<double>> nodegrads(1, std::vector<double>(INPUT_SIZE, 0)); // Gradients d(squared error) / dN'
        for (int i = 0; i < HIDDEN_LAYERS; i++) nodegrads.push_back(std::vector<double>(NODES_PER_HIDDEN, 0));
        
        // nodegrads[i][j] is d(squared error) / d(value' of node j in layer i so after the sigmoid)
        
        NeuralNetwork gradients(*this); // weight gradients. 
        // gradients.weights[l][i][j] is the gradient of the edge starting on node i on layer l leading into node j on layer l + 1
        
        // For the output layer it simply has d(sqerror) / d(output')
        nodegrads.push_back(Eprime);
        // Now for successive layers:
        for (int i = HIDDEN_LAYERS; i >= 0; i--) {
            if (verbose) {
                std::cout << "LAYER " << i << " (DERIVED FROM LAYER " << (i + 1) << ")\n";
                std::cout << "SHAPE OF nodegrads[i + 1] :" << nodegrads[i + 1].size() << " [i] :" << nodegrads[i].size() << "\n";
                std::cout << "SHAPE OF values[i] " << dim(values[i]) << " weights[i] " << dim(weights[i]) << "\n";
                std::cout << "[j] goes up to " << nodegrads[i + 1].size() - 1 << " [k] " << values[i].rows() - 2 << "\n"; 
            }

            for (int k = 0; k < values[i].rows() - 1; k++) { // node k on layer i
                double sum = 0;
                for (int j = 0; j < nodegrads[i + 1].size(); j++) {
                    if (i == HIDDEN_LAYERS) sum += nodegrads[i + 1][j] * finalad(values[i + 1](j)) * weights[i](k, j);
                    else sum += nodegrads[i + 1][j] * activd(values[i + 1](j)) * weights[i](k, j);
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

    // too lazy to do the matmul stuff right now partly because the matmul in the single output version is not good.
    void backprop(std::vector<double> yhat, std::vector<double> y, double alpha, bool verbose = false) {
        backpropsimple(yhat, y, alpha, verbose);
    }
    
    private:

    std::string dim(Eigen::MatrixXd v) {
        return "[" + std::to_string(v.rows()) + " " + std::to_string(v.cols()) + "]";
    }

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
        int closebracket = find(data, ']', space2 + 1);
        int newline = find(data, '\n', closebracket + 1);
        int output = std::stoi(substring(data, closebracket + 1, newline));
        std::cout << input << " " << layers << " " << hidden << " " << output << "\n";
    
        NeuralNetwork nn(input, layers, hidden, output);
    
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
            for (int out = 0; out < output; out++) {
                int end = find(data, ' ', start);
                nn.weights[layers](in, out) = std::stod(substring(data, start, end));
                start = end + 1;
            }
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

NeuralNetwork randomAI(double radius = 1, int protogen = DEFAULT_INPUT, int primagen = DEFAULT_LAYERS, int primogenitor = DEFAULT_HIDDEN, int zenith = DEFAULT_OUTPUT) {
    NeuralNetwork nn(protogen, primagen, primogenitor, zenith);
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


/*

// EXAMPLE CODE



#include <iostream>
#include <ctime>
#include <chrono>
#include <vector>
using namespace std;

double randf() { return (double)(rand()) / (double)(RAND_MAX); }
double randrad() { return 2 * (0.5 - randf()); }

int n = 10000;

void circles(vector<vector<double>> centers = {{8, 4}, {-10, -8}, {4, -12}}, vector<double> radii = {8, 6, 7}) {
    NeuralNetwork nn = Genetic::randomAI(1, centers[0].size(), 2, 10, centers.size() + 1);
    
    cout << nn.toString() << endl;
    
    int good = 0;
    int bad = 0;
    
    for (int i = 0; i < n; i++) {
        vector<double> v(nn.INPUT_SIZE);
        for (int j = 0; j < nn.INPUT_SIZE; j++) v[j] = 16 * randrad();
        double res = 0;
        for (int i = 0; i < centers.size(); i++) {
            double rsq = 0;
            for (int j = 0; j < nn.INPUT_SIZE; j++) rsq += (v[j] - centers[i][j]) * (v[j] - centers[i][j]);
            if (rsq < radii[i] * radii[i]) {
                res = i;
                break;
            }
        }
        
        vector<double> desired(nn.OUTPUT_SIZE, 0);
        desired[res] = 1;
        
        vector<double> yhat = nn.eval(v);
        
        int pickedindex = 0;
        double maximum = DBL_MAX * -1;
        int realindex = res;
        for (int i = 0; i < yhat.size(); i++) {
            if (yhat[i] > maximum) {
                maximum = yhat[i];
                pickedindex = i;
            }
        }
        if (realindex == pickedindex) good++;
        else bad++;

        // std::cout << realindex << " / " << pickedindex << "\n";

        nn.backprop(yhat, desired, 0.01, false);
        
        if (i % (n / 100) == 0 && i > 0) {
            std::cout << "ITERATIONS " << i - (n / 100) << " - " << i << " | " << good << " PASS / " << bad << " FAIL\n";
            good = 0;
            bad = 0;
        }
        
        // std::cout << nn.toString() << "\n";
    }
}

int main()
{
    srand(time(0));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    circles();

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    

    return 0;

    cout << "NEXT\n";
    
    NeuralNetwork nn = Genetic::randomAI(16, 16, 2, 10, 10);
    
    cout << nn.shape() << endl;

	auto v = nn.eval(std::vector<double>(16, 1), true);
    cout << ">>>";
    for (auto i :v )cout << i << " ";
    cout << v.size() << endl;

    return 0;
}

*/

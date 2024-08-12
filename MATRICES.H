#ifndef MATRICES_H
#define MATRICES_H
#include <vector>
#include <iostream>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <string>
#include <cmath>

class Matrix {
    public:
    int n;
    int m;
    
    std::vector<std::vector<double>> mat;
    
    bool operator<(const Matrix& other) {
        return mat < other.mat;
    }
    
    Matrix() {
        n = 4;
        m = 4;
        mat = std::vector<std::vector<double>>(n, std::vector<double>(m, 0));
    }
    
    Matrix(int a, int b) {
        n = a;
        m = b;
        mat = std::vector<std::vector<double>>(n, std::vector<double>(m, 0));
    }
    
    Matrix(std::vector<std::vector<double>> v) {
        n = v.size();
        m = v[0].size();
        mat = std::vector<std::vector<double>>(n, std::vector<double>(m, 0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) mat[i][j] = v[i][j];
        }
    }
    
    Matrix(const Matrix& other) {
        n = other.n;
        m = other.m;
        mat = std::vector<std::vector<double>>(n, std::vector<double>(m, 0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) mat[i][j] = other.mat[i][j];
        }
    }
    
    // Special matrices
    
    static Matrix NIL() {
        return Matrix(0, 0);
    }

    static Matrix eye(int n) {
        Matrix m(n, n);
        for (int i = 0; i < n; i++) m.mat[i][i] = 1;
        return m;
    }
    
    // Multiplying a matrix by this one ON THE LEFT SIDE swaps rows a and b in the original
    static Matrix swap(int n, int a, int b) {
        Matrix m = eye(n);
        for (int i = 0; i < n; i++) {
            double temp = m.mat[a][i];
            m.mat[a][i] = m.mat[b][i];
            m.mat[b][i] = temp;
        }
        return m;
    }
    
    // Multiplying a matrix by this one adds v times r1 to r2
    static Matrix add(int n, int r1, int r2, int v) {
        Matrix m = eye(n);
        m.mat[r2][r1] = v;
        return m;
    }
    
    // Properties
    
    bool isSquare() {
        return n == m;
    }
    
    bool isRow() {
        return n == 1;
    }
    
    bool isColumn() {
        return m == 1;
    }
    
    bool isNull() {
        return n == 0 && m == 0;
    }
    
    bool isZero() {
        for (auto i : mat) {
            for (auto j : i) {
                if (j != 0) return false;
            }
        }
        return true;
    }
    
    // Remove rows and columns
    
    Matrix remrow(int desired) {
        if (desired < 0 || desired >= n) return NIL();
        Matrix res(n - 1, m);
        int row = 0;
        for (int i = 0; i < n; i++) {
            if (i == desired) continue;
            for (int j = 0; j < m; j++) res.mat[row][j] = mat[i][j];
            row++;
        }
        return res;
    }
    
    Matrix remcol(int desired) {
        if (desired < 0 || desired >= m) return NIL();
        Matrix res(n, m - 1);
        int col = 0;
        for (int i = 0; i < m; i++) {
            if (i == desired) continue;
            for (int j = 0; j < n; j++) res.mat[j][col] = mat[j][i];
            col++;
        }
        return res;
    }
    
    // Operations on matrices
    
    Matrix matmul(const Matrix& other) {
        if (m != other.n) return NIL();
        Matrix res(n, other.m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < other.m; j++) {
                for (int k = 0; k < m; k++) res.mat[i][j] += mat[i][k] * other.mat[k][j];
            }
        }
        return res;
    }
    
    Matrix add(const Matrix& other) {
        int nx = std::min(n, other.n);
        int mx = std::min(m, other.m);
        Matrix res(nx, mx);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < mx; j++) res.mat[i][j] = mat[i][j] + other.mat[i][j];
        }
        return res;
    }
    
    Matrix sub(const Matrix& other) {
        int nx = std::min(n, other.n);
        int mx = std::min(m, other.m);
        Matrix res(nx, mx);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < mx; j++) res.mat[i][j] = mat[i][j] - other.mat[i][j];
        }
        return res;
    }
    
    Matrix operator+(const Matrix& other) {
        return add(other);
    }
    
    Matrix operator-(const Matrix& other) {
        return sub(other);
    }
    
    Matrix operator*(const Matrix& other) {
        return matmul(other);
    }
    
    Matrix operator*(const double& other) {
        Matrix res(*this);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res.mat[i][j] *= other;
        }
        return res;
    }
    
    Matrix transpose() {
        Matrix res(m, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res.mat[j][i] = mat[i][j];
        }
        return res;
    }
    
    Matrix ref() {
        Matrix res(*this);
        int h = 0;
        int k = 0;
        while (h < n && k < m) {
            int mrow = -1;
            double mbeep = 0;
            for (int i = h; i < n; i++) {
                double test = std::abs(res.mat[i][k]);
                if (test > mbeep) {
                    mbeep = test;
                    mrow = i;
                }
            }
            if (mbeep == 0) {
                k++;
                continue;
            }
            
            if (mrow != h) res = swap(n, mrow, h) * res;
            
            for (int i = h + 1; i < n; i++) {
                double val = res.mat[i][k] / res.mat[h][k];
                res.mat[i][k] = 0;
                for (int j = k + 1; j < n; j++) res.mat[i][j] -= val * res.mat[h][j];
            }
            
            h++;
            k++;
        }
        return res;
    }
    
    double det() {
        if (n != m) return 0;
        
        double detscal = 1;
        
        Matrix res(*this);
        int h = 0;
        int k = 0;
        while (h < n && k < m) {
            res.disp();
            int mrow = -1;
            double mbeep = 0;
            for (int i = h; i < n; i++) {
                double test = std::abs(res.mat[i][k]);
                if (test > mbeep) {
                    mbeep = test;
                    mrow = i;
                }
            }
            if (mbeep == 0) {
                k++;
                continue;
            }
            
            if (mrow != h) {
                detscal *= -1;
                res = swap(n, mrow, h) * res;
            }
            
            for (int i = h + 1; i < n; i++) {
                double val = res.mat[i][k] / res.mat[h][k];
                res.mat[i][k] = 0;
                for (int j = k + 1; j < n; j++) res.mat[i][j] -= val * res.mat[h][j];
            }
            
            h++;
            k++;
        }
        for (int i = 0; i < n; i++) detscal *= res.mat[i][i];
        return detscal;
    }
    
    Matrix inverse() {
        if (n != m) return NIL();
        Matrix res(*this);
        Matrix inv = eye(n);
        int h = 0;
        int k = 0;
        while (h < n && k < m) {
            int mrow = -1;
            double mbeep = 0;
            for (int i = h; i < n; i++) {
                double test = std::abs(res.mat[i][k]);
                if (test > mbeep) {
                    mbeep = test;
                    mrow = i;
                }
            }
            if (mbeep == 0) {
                k++;
                continue;
            }
            /*
            for (int j = 0; j < m; j++) {
                double temp = res.mat[mrow][j];
                res.mat[mrow][j] = res.mat[h][j];
                res.mat[h][j] = temp;
                
                temp = inv.mat[mrow][j];
                inv.mat[mrow][j] = inv.mat[h][j];
                inv.mat[h][j] = temp;
            }
            */
            
            res = swap(n, mrow, h) * res;
            inv = swap(n, mrow, h) * inv;
            
            for (int i = h + 1; i < n; i++) {
                double val = res.mat[i][k] / res.mat[h][k];
                res.mat[i][k] = 0;
                for (int j = k + 1; j < n; j++) res.mat[i][j] -= val * res.mat[h][j];
                for (int j = 0; j < n; j++) inv.mat[i][j] -= val * inv.mat[h][j];
            }
            
            
            h++;
            k++;
        }
        
        // Upper triangular --> diagonal
        
        for (int i = 0; i < n; i++) {
            if (res.mat[i][i] == 0) return NIL();
        }
        
        for (h = 1; h < n; h++) { // here h represents the row that we are pivoting off of but also the column we check for 0s in
            for (int i = 0; i < h; i++) {
                if (res.mat[i][h] == 0) continue;
                double val = res.mat[i][h] / res.mat[h][h];
                for (int j = 0; j < m; j++) {
                    res.mat[i][j] -= val * res.mat[h][j];
                    inv.mat[i][j] -= val * inv.mat[h][j];
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) inv.mat[i][j] /= res.mat[i][i];
        }
        
        return inv;
    }
    
    // DECOMPOSITIONS AND FACTORIZATIONS
    
    // Decomposes a matrix into the following equation PA = LU. The "real" factorization is A = (P')LU
    std::vector<Matrix> PLU() {
        if (n != m) return {NIL(), NIL(), NIL()};
        
        Matrix res(*this);
        Matrix L = Matrix(n, m);
        Matrix P = eye(n);
        int h = 0;
        int k = 0;
        while (h < n && k < m) {
            int mrow = -1;
            double mbeep = 0;
            for (int i = h; i < n; i++) {
                double test = std::abs(res.mat[i][k]);
                if (test > mbeep) {
                    mbeep = test;
                    mrow = i;
                }
            }
            if (mbeep == 0) {
                k++;
                continue;
            }
            
            if (mrow != h) {
                P = swap(n, mrow, h) * P;
                L = swap(n, mrow, h) * L;
                res = swap(n, mrow, h) * res;
            }
            
            for (int i = h + 1; i < n; i++) {
                double val = res.mat[i][k] / res.mat[h][k];
                res.mat[i][k] = 0;
                for (int j = k + 1; j < n; j++) res.mat[i][j] -= val * res.mat[h][j];
                // If you are subtracting multiples of row h from row i
                // the affected element in L is row i column h
                L.mat[i][h] = val;
            }
            
            h++;
            k++;
        }
        
        for (int i = 0; i < n; i++) L.mat[i][i] = 1;
        return std::vector<Matrix>{P, L, res};
    }
    
    // QR decomposition
    
    std::pair<Matrix, Matrix> QR() {
        if (n != m) return {NIL(), NIL()};
        
        Matrix q = gramschmidt();
        Matrix r(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) r.mat[i][j] = (q.col(i)).cdot(col(j));
        }
        
        
        return {q, r};
    }
    
    std::pair<Matrix, Matrix> qr() {
        return QR();
    }
    
    /*
    
    // QR decomposition using householder transforms
    std::pair<Matrix, Matrix> householderQR() {
        if (n != m) return {NIL(), NIL()};
        std::vector<Matrix> minors;
        std::vector<Matrix> qprimes;
        std::vector<Matrix> qs;
        minors.push_back(Matrix(*this));
        
        for (int i = 0; i < n - 1; i++) {
            Matrix x = minors[i].col(0);
            double alpha = x.cnorm();
            Matrix e = Matrix(n - i, 1);
            e.mat[0][0] = 1;
            
            Matrix u = x - (e * alpha);
            Matrix v = u.unit();
            Matrix QP = eye(n) - ((v * (v.transpose())) * 2);
            
            qprimes.push_back(QP);
            
            Matrix Q = eye(n);
            Q.emplace(QP, i, i);
            qs.push_back(Q);
            
            Matrix hi = Q * minors[0];
            
            std::cout << "HOUSEHOLDER."<< i << "\n";
            x.disp();
            u.disp();
            QP.disp();
            Q.disp();
            hi.disp();
            
            hi = hi.remrow(0);
            hi = hi.remcol(0);
            minors.push_back(hi);
        }
        
        Matrix q = eye(n);
        for (auto i : qs) q = q * (i.transpose());
        
        Matrix r = (q.transpose()) * minors[0];
        return {q, r};
    }
    
    */
    
    // Schur decomposition: repeated application of the QR decomposition decomposes
    // A = QR(Q') where Q is the unitary matrix and R is block upper triangular
    // Q contains eigenvectors and R contains, for each 1x1 block, a real eigenvalue, and for each 2x2 block, a pair of conjugate eigenvalues.
    std::pair<Matrix, Matrix> schur(int iterations = 16) {
        Matrix A(*this);
        std::pair<Matrix, Matrix> p;
        for (int i = 0; i < iterations; i++) {
            p = A.QR();
            A = p.second * p.first;
            p.first.disp();
            p.second.disp();
        }
        return A.QR();
    }
    
    Matrix eigenvector(double ev) {
        Matrix A = (eye(n) * ev) - Matrix(*this);
        A = A.ref();
        
        Matrix vec(m, 1);
        for (int i = m - 1; i >= 0; i--) {
            
        }
    }
    
    // Gram-Schmidt process to produce a matrix that has an orthonormal basis for the column space of the original
    
    Matrix gramschmidt() {
        Matrix res(n, m);
        int index = 0;
        for (int i = 0; i < m; i++) {
            Matrix crazyprojection = Matrix(n, 1);
            Matrix column = col(i);
            for (int j = 0; j < i; j++) crazyprojection = crazyprojection + column.cproj(res.col(j));
            Matrix incoming = column - crazyprojection;
            if (incoming.isZero()) continue;
            res.implant(incoming.unit(), index++);
        }
        return res;
    }
    
    // MATRICES AND VECTORS
    
    // Grab a column as an individual vector
    Matrix col(int i) { 
        Matrix res(n, 1);
        for (int r = 0; r < n; r++) res.mat[r][0] = mat[r][i];
        return res;
    }
    
    // Grab a row as an individual vector
    Matrix row(int i) { 
        Matrix res(1, n);
        for (int r = 0; r < m; r++) res.mat[0][r] = mat[i][r];
        return res;
    }
    
    // Emplace a column vector as a matrix column (or a submatrix starting from that column)
    void implant(Matrix other, int s) {
        for (int i = 0; i < other.n && i < n; i++) {
            for (int j = s; j < s + other.m && j < m; j++) mat[i][j] = other.mat[i][j - s];
        }
    }
    
    // Emplace a matrix as a submatrix
    
    void emplace(Matrix other, int sr, int sc) {
        for (int i = sr; i < sr + other.n && i < n; i++) {
            for (int j = sc; j < sc + other.m && j < m; j++) mat[i][j] = other.mat[i - sr][j - sc];
        }
    }
    
    // Operations on column vectors.
    // The dot product is A cdot B = transpose(A) * B. If the matrices are larger we take the [0][0] element.
    double cdot(Matrix other) {
        return (transpose() * other).mat[0][0];
    }
    
    double crsq() {
        return cdot(*this);
    }
    
    double cnorm() {
        return std::sqrt(crsq());
    }
    
    Matrix cproj(Matrix other) { // Projects (this) onto the direction of (other)
        return other * (cdot(other) / other.crsq());
    }
    
    Matrix unit() {
        return Matrix(*this) * (1.0 / cnorm());
    }
    
    // Printouts and String Representations
    
    std::string format(double d, int L = 8) {
        std::string res = std::to_string(d);
        while (res.length() < L) res = res + "0";
        return res.substr(0, L);
    }
    
    std::string toString() {
        if (isNull()) return "[NULL]";
        std::string res = "[" + std::to_string(n) + " " + std::to_string(m) + "]\n";
        for (int i = 0; i < n; i++) {
            res = res + "[ ";
            for (int j = 0; j < m; j++) res = res + format(mat[i][j]) + " ";
            res = res + "]\n";
        }
        return res;
    }
    
    void disp() {
        std::cout << toString();
    }
    
    // Miscellaneous
    
    static Matrix random(int n, int m) {
        Matrix res(n, m);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) res.mat[i][j] = (double)(rand()) / (double)(RAND_MAX);
        }
        return res;
    }
};

#endif

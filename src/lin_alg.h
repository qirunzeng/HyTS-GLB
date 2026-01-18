#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

struct Vec {
    std::vector<double> a;

    explicit Vec(int d=0, double v=0.0) : a(d, v) {}
    int dim() const { return (int)a.size(); }

    double& operator[](int i) { return a[i]; }
    double  operator[](int i) const { return a[i]; }

    double norm2() const {
        double s = 0.0;
        for (double x: a) s += x*x;
        return std::sqrt(s);
    }
};

inline Vec operator+(const Vec& x, const Vec& y) {
    if (x.dim()!=y.dim()) throw std::runtime_error("Vec dim mismatch");
    Vec z(x.dim());
    for (int i=0;i<x.dim();++i) z[i]=x[i]+y[i];
    return z;
}
inline Vec operator-(const Vec& x, const Vec& y) {
    if (x.dim()!=y.dim()) throw std::runtime_error("Vec dim mismatch");
    Vec z(x.dim());
    for (int i=0;i<x.dim();++i) z[i]=x[i]-y[i];
    return z;
}
inline Vec operator*(double c, const Vec& x) {
    Vec z(x.dim());
    for (int i=0;i<x.dim();++i) z[i]=c*x[i];
    return z;
}
inline double dot(const Vec& x, const Vec& y) {
    if (x.dim()!=y.dim()) throw std::runtime_error("Vec dim mismatch");
    double s=0.0;
    for (int i = 0; i < x.dim(); ++i) {
        s += x[i] * y[i];
    }
    return s;
}

struct Mat {
    int n;                 // square matrix n x n
    std::vector<double> a; // row-major

    explicit Mat(int n_=0, double diag=0.0) : n(n_), a(n_*n_, 0.0) {
        if (diag!=0.0) {
            for (int i=0;i<n;++i) (*this)(i,i)=diag;
        }
    }
    double& operator()(int i,int j) { return a[i*n + j]; }
    double  operator()(int i,int j) const { return a[i*n + j]; }

    static Mat identity(int n) { return Mat(n, 1.0); }
};

inline Mat operator+(const Mat& A, const Mat& B) {
    if (A.n!=B.n) throw std::runtime_error("Mat dim mismatch");
    Mat C(A.n);
    for (int i=0;i<A.n*A.n;++i) C.a[i]=A.a[i]+B.a[i];
    return C;
}
inline Mat operator*(double c, const Mat& A) {
    Mat B(A.n);
    for (int i=0;i<A.n*A.n;++i) B.a[i]=c*A.a[i];
    return B;
}

inline Vec mat_vec(const Mat& A, const Vec& x) {
    if (A.n!=x.dim()) throw std::runtime_error("Mat-Vec dim mismatch");
    Vec y(A.n);
    for (int i=0;i<A.n;++i) {
        double s=0.0;
        for (int j=0;j<A.n;++j) s += A(i,j)*x[j];
        y[i]=s;
    }
    return y;
}

inline Mat outer(const Vec& x) {
    int d=x.dim();
    Mat A(d);
    for (int i=0;i<d;++i) for (int j=0;j<d;++j) A(i,j)=x[i]*x[j];
    return A;
}

// Gauss-Jordan inverse with partial pivoting (O(d^3)), for small/medium d.
inline Mat inverse(Mat A) {
    int n=A.n;
    Mat I = Mat::identity(n);

    for (int col=0; col<n; ++col) {
        int piv=col;
        double best = std::fabs(A(col,col));
        for (int r=col+1;r<n;++r) {
            double v = std::fabs(A(r,col));
            if (v>best) { best=v; piv=r; }
        }
        if (best < 1e-12) throw std::runtime_error("Matrix singular / ill-conditioned");

        if (piv!=col) {
            for (int j=0;j<n;++j) std::swap(A(col,j), A(piv,j));
            for (int j=0;j<n;++j) std::swap(I(col,j), I(piv,j));
        }

        double diag = A(col,col);
        for (int j=0;j<n;++j) { A(col,j) /= diag; I(col,j) /= diag; }

        for (int r=0;r<n;++r) if (r!=col) {
            double f = A(r,col);
            if (f==0.0) continue;
            for (int j=0;j<n;++j) {
                A(r,j) -= f*A(col,j);
                I(r,j) -= f*I(col,j);
            }
        }
    }
    return I;
}

inline double quad_form(const Mat& Ainv, const Vec& g) {
    // g^T Ainv g
    Vec t = mat_vec(Ainv, g);
    return dot(g, t);
}


inline Vec solve_spd_cholesky(const Mat& A, const Vec& b) {
    int n = A.n;
    std::vector<double> L(n*n, 0.0);

    auto Lij = [&](int i,int j)->double& { return L[i*n+j]; };
    auto Lijc = [&](int i,int j)->double { return L[i*n+j]; };

    // factorize A = L L^T
    for (int i=0;i<n;++i) {
        for (int j=0;j<=i;++j) {
            double s = A(i,j);
            for (int k=0;k<j;++k) s -= Lijc(i,k) * Lijc(j,k);
            if (i==j) {
                if (s <= 1e-14) throw std::runtime_error("Cholesky failed: not SPD / ill-conditioned");
                Lij(i,j) = std::sqrt(s);
            } else {
                Lij(i,j) = s / Lijc(j,j);
            }
        }
    }

    // forward: L y = b
    Vec y(n, 0.0);
    for (int i=0;i<n;++i) {
        double s = b[i];
        for (int k=0;k<i;++k) s -= Lijc(i,k) * y[k];
        y[i] = s / Lijc(i,i);
    }

    // backward: L^T x = y
    Vec x(n, 0.0);
    for (int i=n-1;i>=0;--i) {
        double s = y[i];
        for (int k=i+1;k<n;++k) s -= Lijc(k,i) * x[k];
        x[i] = s / Lijc(i,i);
    }
    return x;
}

inline double quad_form_inv_spd(const Mat& A, const Vec& g) {
    Vec x = solve_spd_cholesky(A, g);
    return dot(g, x);
}

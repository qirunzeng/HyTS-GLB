#pragma once
#include "lin_alg.h"

struct Chol {
    int n;
    double *L;

    explicit Chol(int n_=0) : n(n_), L(nullptr) {
        if (n_ > 0) {
            L = new double[n_*n_];
            std::fill(L, L + n_*n_, 0.0);
        }
    }

    ~Chol() { delete[] L; }

    Chol(const Chol&) = delete;
    Chol& operator=(const Chol&) = delete;

    Chol(Chol&& other) noexcept : n(other.n), L(other.L) {
        other.L = nullptr;
        other.n = 0;
    }

    Chol& operator=(Chol&& other) noexcept {
        if (this != &other) {
            delete[] L;
            n = other.n;
            L = other.L;
            other.L = nullptr;
            other.n = 0;
        }
        return *this;
    }

    inline double& at(int i, int j) {
        return L[i*n + j];
    }

    inline const double& at(int i, int j) const {
        return L[i*n + j];
    }
};

inline Chol chol_spd(const Mat& A) {
    int n = A.n;
    Chol C(n);

    double* L = C.L;

    for (int i = 0; i < n; ++i) {
        double* Li = L + i*n;

        for (int j = 0; j <= i; ++j) {
            double* Lj = L + j*n;

            double s = A(i,j);

            for (int k = 0; k < j; ++k) {
                s -= Li[k] * Lj[k];
            }

            if (i == j) {
                if (s <= 0.0)  // better than 1e-14
                    throw std::runtime_error("Cholesky failed: not SPD");

                Li[j] = std::sqrt(s);
            } else {
                Li[j] = s / Lj[j];
            }
        }
    }

    return C;
}

inline Vec solve_chol(const Chol& C, const Vec& b) {
    int n = C.n;
    if (b.dim() != n) throw std::runtime_error("solve_chol dim mismatch");

    Vec y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= C.at(i,k) * y[k];
        y[i] = s / C.at(i,i);
    }

    Vec x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= C.at(k,i) * x[k];
        x[i] = s / C.at(i,i);
    }
    return x;
}

inline double quad_form_inv_chol(const Chol& C, const Vec& g) {
    int n = C.n;
    if (g.dim() != n) throw std::runtime_error("quad_form_inv_chol dim mismatch");

    Vec y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = g[i];
        for (int k = 0; k < i; ++k) s -= C.at(i,k) * y[k];
        y[i] = s / C.at(i,i);
    }
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += y[i] * y[i];
    return ss;
}

#include <iostream>
#include "jojograd/variable.hpp"

// Demonstrates gradient flow through a multi-step expression.
//
// Expression:  f = (a * b) / b
// Mathematically simplifies to a, so df/da should be 1 and df/db should be 0.
// The graph does NOT simplify symbolically — gradients cancel numerically
// during backprop, which is exactly what we verify here.
int main() {
    jojo::variable<double> a(10.0);
    jojo::variable<double> b(2.0);

    jojo::variable<double> c = a * b;   // c = 20
    jojo::variable<double> d = c / b;   // d = 10

    d.backward(1.0);

    std::cout << "Expression: f = (a * b) / b\n";
    std::cout << "  a=" << a.data << "  b=" << b.data << "\n";
    std::cout << "  f=" << d.data << "\n";
    std::cout << "  df/da=" << a.grad << "  (expected 1)\n";
    std::cout << "  df/db=" << b.grad << "  (expected 0)\n";
    std::cout << "  df/dc=" << c.grad << "  (expected 0.5)\n";

    return 0;
}

#include <iostream>
#include "jojograd/variable.hpp"

// Demonstrates forward values and gradients for each arithmetic operation.
int main() {
    jojo::variable<double> a(6.0);
    jojo::variable<double> b(3.0);

    {
        // f = a + b  =>  df/da = 1,  df/db = 1
        jojo::variable<double> f = a + b;
        f.backward(1.0);
        std::cout << "a + b:  f=" << f.data
                  << "  da=" << a.grad << "  db=" << b.grad << "\n";
        a.grad = 0; b.grad = 0;
    }

    {
        // f = a * b  =>  df/da = b = 3,  df/db = a = 6
        jojo::variable<double> f = a * b;
        f.backward(1.0);
        std::cout << "a * b:  f=" << f.data
                  << "  da=" << a.grad << "  db=" << b.grad << "\n";
        a.grad = 0; b.grad = 0;
    }

    {
        // f = a - b  =>  df/da = 1,  df/db = -1
        jojo::variable<double> f = a - b;
        f.backward(1.0);
        std::cout << "a - b:  f=" << f.data
                  << "  da=" << a.grad << "  db=" << b.grad << "\n";
        a.grad = 0; b.grad = 0;
    }

    {
        // f = a / b  =>  df/da = 1/b = 0.333,  df/db = -a/b² = -0.666
        jojo::variable<double> f = a / b;
        f.backward(1.0);
        std::cout << "a / b:  f=" << f.data
                  << "  da=" << a.grad << "  db=" << b.grad << "\n";
        a.grad = 0; b.grad = 0;
    }

    {
        // f = -a  =>  df/da = -1
        jojo::variable<double> f = -a;
        f.backward(1.0);
        std::cout << "  -a:  f=" << f.data
                  << "  da=" << a.grad << "\n";
        a.grad = 0;
    }

    return 0;
}

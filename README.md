# JojoGrad

A minimal C++ library for reverse-mode automatic differentiation.

Variables form a computational graph as you apply arithmetic operations on them.
Calling `backward()` on any node walks the graph in reverse and accumulates
partial derivatives (gradients) into every upstream variable via the chain rule.

## Build

```bash
cmake -B build
cmake --build build
```

Requires CMake ≥ 3.14 and a C++17 compiler. Tests are fetched and built automatically via FetchContent (GoogleTest v1.14).

## Run tests

```bash
cd build && ctest --output-on-failure
```

## Run examples

```bash
./build/example_basic_ops
./build/example_chain_rule
```

## Usage

```cpp
#include "jojograd/variable.hpp"

jojo::variable<double> a(3.0);
jojo::variable<double> b(4.0);

jojo::variable<double> c = a * b;
c.backward(1.0);

// a.grad == 4.0  (dc/da = b)
// b.grad == 3.0  (dc/db = a)
```

Operands must outlive any result derived from them — the graph holds raw pointers.

## Supported types

Explicit instantiations for `double` and `float` are provided out of the box.
To use another numeric type, add this in one of your `.cpp` files:

```cpp
template class jojo::variable<MyType>;
```

## Supported operations

`+`  `*`  `-`  `/`  unary `-`

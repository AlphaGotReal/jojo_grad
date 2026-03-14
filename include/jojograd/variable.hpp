#pragma once

#include <vector>
#include <stdexcept>

namespace jojo {

class invalid_operation : public std::exception {
// Thrown when an operation token has no defined gradient rule.
public:
  const char* what() const noexcept override;
  };

  // Encodes which binary arithmetic operation produced a variable node.
  // Tokens: 0=+  1=*  2=-  3=/  -1=none (leaf node)
  struct operation {
  int token;

  operation();
  explicit operation(int t);
};

template<typename T>
class variable {
// A node in the reverse-mode autodiff computational graph.
//
// Forward pass: arithmetic operators build the graph by linking result nodes
// back to their operand nodes via raw pointers (children).
//
// Backward pass: backward() walks the graph in reverse, accumulating
// partial derivatives into each node's grad field via the chain rule.
//
// Lifetime: children holds raw pointers to operands, so operands must
// outlive any result derived from them.
public:
  T      data;
  double grad;

  std::vector<variable<T>*> children; // operand nodes that produced this node
  operation op;                        // operation applied to children

  variable();
  explicit variable(T data);

  variable operator+(variable<T>& var);
  variable operator*(variable<T>& var);
  variable operator-(variable<T>& var);
  variable operator/(variable<T>& var);
  variable operator-();

  void backward(double gradient);
};

// Explicit instantiations are provided for double and float in variable.cpp.
// To use another numeric type, add an explicit instantiation in your own
// translation unit: template class jojo::variable<YourType>;
extern template class variable<double>;
extern template class variable<float>;

} // namespace jojo

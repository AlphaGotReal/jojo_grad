#include "jojograd/variable.hpp"

namespace jojo {

// ---- invalid_operation ----

const char* invalid_operation::what() const noexcept {
  return "invalid operation performed";
}

// ---- operation ----

operation::operation() : token(-1) {}
operation::operation(int t) : token(t) {}

// ---- variable ----

template<typename T>
variable<T>::variable() : data((T)0), grad(0.0) {}

template<typename T>
variable<T>::variable(T data) : data(data), grad(0.0) {}

template<typename T>
variable<T> variable<T>::operator+(variable<T>& var) {
  variable<T> res(data + var.data);
  res.children.push_back(this);
  res.children.push_back(&var);
  res.op.token = 0;
  return res;
}

template<typename T>
variable<T> variable<T>::operator*(variable<T>& var) {
  variable<T> res(data * var.data);
  res.children.push_back(this);
  res.children.push_back(&var);
  res.op.token = 1;
  return res;
}

template<typename T>
variable<T> variable<T>::operator-(variable<T>& var) {
  variable<T> res(data - var.data);
  res.children.push_back(this);
  res.children.push_back(&var);
  res.op.token = 2;
  return res;
}

template<typename T>
variable<T> variable<T>::operator/(variable<T>& var) {
  variable<T> res(data / var.data);
  res.children.push_back(this);
  res.children.push_back(&var);
  res.op.token = 3;
  return res;
}

template<typename T>
variable<T> variable<T>::operator-() {
  variable<T> res(-data);
  res.children.push_back(this);
  res.op.token = 2;
  return res;
}

template<typename T>
void variable<T>::backward(double gradient) {
  grad += gradient;

  if (children.empty()) return;

  // Unary case: only negation (-x) reaches here with one child.
  if (children.size() == 1) {
    double upstream = (op.token == 2) ? -grad : grad;
    children[0]->backward(upstream);
    return;
  }

  // Binary cases — standard chain rule for each supported operation.
  switch (op.token) {
  case 0: // d(a+b)/da = 1,  d(a+b)/db = 1
    children[0]->backward(grad);
    children[1]->backward(grad);
    break;

  case 1: // d(a*b)/da = b,  d(a*b)/db = a
    children[0]->backward(grad * static_cast<double>(children[1]->data));
    children[1]->backward(grad * static_cast<double>(children[0]->data));
    break;

  case 2: // d(a-b)/da = 1,  d(a-b)/db = -1
    children[0]->backward(grad);
    children[1]->backward(-grad);
    break;

  case 3: { // d(a/b)/da = 1/b,  d(a/b)/db = -a/b²
    if (children[1]->data == (T)0)
      throw std::runtime_error("division by zero in backward pass");

    double b  = static_cast<double>(children[1]->data);
    double a  = static_cast<double>(children[0]->data);
    children[0]->backward(grad / b);
    children[1]->backward(-grad * a / (b * b));
    break;
  }

  default:
    throw invalid_operation();
  }
}

// Explicit instantiations — these make the template definitions in this
// translation unit available to the linker for double and float.
template class variable<double>;
template class variable<float>;

} // namespace jojo

#include <gtest/gtest.h>
#include "jojograd/variable.hpp"

// ---- Forward values ----

TEST(ForwardPass, Addition) {
  jojo::variable<double> a(3.0), b(4.0);
  jojo::variable<double> c = a + b;
  EXPECT_DOUBLE_EQ(c.data, 7.0);
}

TEST(ForwardPass, Multiplication) {
  jojo::variable<double> a(3.0), b(4.0);
  jojo::variable<double> c = a * b;
  EXPECT_DOUBLE_EQ(c.data, 12.0);
}

TEST(ForwardPass, Subtraction) {
  jojo::variable<double> a(5.0), b(2.0);
  jojo::variable<double> c = a - b;
  EXPECT_DOUBLE_EQ(c.data, 3.0);
}

TEST(ForwardPass, Division) {
  jojo::variable<double> a(6.0), b(2.0);
  jojo::variable<double> c = a / b;
  EXPECT_DOUBLE_EQ(c.data, 3.0);
}

TEST(ForwardPass, UnaryNegation) {
  jojo::variable<double> a(5.0);
  jojo::variable<double> c = -a;
  EXPECT_DOUBLE_EQ(c.data, -5.0);
}

// ---- Gradients ----

TEST(Backward, Addition) {
  jojo::variable<double> a(3.0), b(4.0);
  jojo::variable<double> c = a + b;
  c.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad, 1.0);
  EXPECT_DOUBLE_EQ(b.grad, 1.0);
}

TEST(Backward, Multiplication) {
  jojo::variable<double> a(3.0), b(4.0);
  jojo::variable<double> c = a * b;
  c.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad, 4.0); // df/da = b
  EXPECT_DOUBLE_EQ(b.grad, 3.0); // df/db = a
}

TEST(Backward, Subtraction) {
  jojo::variable<double> a(5.0), b(2.0);
  jojo::variable<double> c = a - b;
  c.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad,  1.0);
  EXPECT_DOUBLE_EQ(b.grad, -1.0);
}

TEST(Backward, Division) {
  jojo::variable<double> a(6.0), b(2.0);
  jojo::variable<double> c = a / b;
  c.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad,  0.5);   // df/da = 1/b
  EXPECT_DOUBLE_EQ(b.grad, -1.5);   // df/db = -a/b²
}

TEST(Backward, UnaryNegation) {
  jojo::variable<double> a(5.0);
  jojo::variable<double> c = -a;
  c.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad, -1.0);
}

TEST(Backward, ChainRule) {
  // f = (a * b) / b  => df/da = 1, df/db = 0 (gradients cancel)
  jojo::variable<double> a(10.0), b(2.0);
  jojo::variable<double> c = a * b;
  jojo::variable<double> d = c / b;
  d.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad, 1.0);
  EXPECT_DOUBLE_EQ(b.grad, 0.0);
}

TEST(Backward, GradientAccumulation) {
  // A node used in two branches accumulates gradients from both paths.
  // f = a + a  => df/da = 2
  jojo::variable<double> a(3.0);
  jojo::variable<double> f = a + a;
  f.backward(1.0);
  EXPECT_DOUBLE_EQ(a.grad, 2.0);
}

TEST(Backward, ScaledGradient) {
  // Passing a non-unit seed propagates the scaling downstream.
  jojo::variable<double> a(3.0), b(4.0);
  jojo::variable<double> c = a + b;
  c.backward(2.0);
  EXPECT_DOUBLE_EQ(a.grad, 2.0);
  EXPECT_DOUBLE_EQ(b.grad, 2.0);
}

// ---- Edge cases ----

TEST(EdgeCases, DivisionByZeroThrows) {
  jojo::variable<double> a(6.0), b(0.0);
  jojo::variable<double> c = a / b;
  EXPECT_THROW(c.backward(1.0), std::runtime_error);
}

TEST(EdgeCases, LeafGradIsZeroInitially) {
  jojo::variable<double> a(5.0);
  EXPECT_DOUBLE_EQ(a.grad, 0.0);
}

TEST(EdgeCases, DefaultConstructedDataIsZero) {
  jojo::variable<double> a;
  EXPECT_DOUBLE_EQ(a.data, 0.0);
}

TEST(EdgeCases, FloatInstantiation) {
  jojo::variable<float> a(2.0f), b(3.0f);
  jojo::variable<float> c = a * b;
  c.backward(1.0);
  EXPECT_FLOAT_EQ(a.grad, 3.0f);
  EXPECT_FLOAT_EQ(b.grad, 2.0f);
}

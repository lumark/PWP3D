#pragma once

namespace PerseusLib
{
namespace Primitives
{
template <typename T>
class Vector2D
{
public:
  T x, y;

  const Vector2D& operator+= (Vector2D &p){
    x += p.x; y += p.y;
    return *this;
  }

  const Vector2D& operator-= (Vector2D &p){
    x -= p.x; y -= p.y;
    return *this;
  }

  Vector2D operator* (T f) const{
    Vector2D r(x * f, y * f);
    return r;
  }

  Vector2D& operator*= (T f){
    x *= f; y *= f;
    return *this;
  }

  friend Vector2D operator+ (const Vector2D &lp, const Vector2D &rp){
    Vector2D r = Vector2D(lp.x + rp.x, lp.y + rp.y);
    return r;
  }

  friend Vector2D operator- (const Vector2D &lp, const Vector2D &rp){
    Vector2D r = Vector2D(lp.x - rp.x, lp.y - rp.y);
    return r;
  }

  friend Vector2D operator& (const Vector2D &a, const Vector2D &b){
    //cross product
    Vector2D r = Vector2D(a.y * b.x, a.x * b.y);
    return r;
  }

  friend T operator| (const Vector2D &a, const Vector2D &b){
    //dot product
    return (T) (a.x*b.x + a.y*b.y);
  }

  friend bool operator== (const Vector2D &lp, const Vector2D &rp){
    return ((lp.x == rp.x) && (lp.y == rp.y));
  }

  //double* ToDoubleArray() { double* a = new double[2]; a[0] = x; a[1] = y; return a; }
  //void FromDoubleArray(double *a) { x = a[0]; y = a[1]; }

  Vector2D(long double x, long double y) { this->x = x; this->y = y; }
  Vector2D(double x, double y) { this->x = x; this->y = y; }
  Vector2D(float x, float y) { this->x = x; this->y = y; }
  Vector2D(int x, int y) { this->x = x; this->y = y; }

  Vector2D(double* d) { this->x = d[0]; this->y = d[0]; }
  Vector2D(float *f) { this->x = f[0]; this->y = f[0]; }
  Vector2D(int *i) { this->x = i[0]; this->y = i[0]; }

  Vector2D(Vector2D *v) { this->x = v->x; this->y = v->y; }
  Vector2D(void) {x = 0; y = 0; }

  ~Vector2D(void) {}
};
}
}

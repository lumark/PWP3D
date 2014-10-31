#pragma once

namespace PerseusLib
{
namespace Primitives
{
template <typename T>
class Vector3D
{
public:
  T x, y, z;

  const Vector3D& operator+= (Vector3D &p){
    x += p.x; y += p.y; z += p.z;
    return *this;
  }

  const Vector3D& operator-= (Vector3D &p){
    x -= p.x; y -= p.y; z -= p.z;
    return *this;
  }

  Vector3D operator* (T f) const{
    Vector3D r(x * f, y * f, z * f);
    return r;
  }

  Vector3D& operator*= (T f){
    x *= f; y *= f; z *= f;
    return *this;
  }

  friend Vector3D operator+ (const Vector3D &lp, const Vector3D &rp){
    Vector3D r = Vector3D(lp.x + rp.x, lp.y + rp.y, lp.z + rp.z);
    return r;
  }

  friend bool operator== (const Vector3D &lp, const Vector3D &rp){
    return ((lp.x == rp.x) && (lp.y == rp.y) && (lp.z == rp.z));
  }

  friend Vector3D operator- (const Vector3D &lp, const Vector3D &rp){
    Vector3D r = Vector3D(lp.x - rp.x, lp.y - rp.y, lp.z - rp.z);
    return r;
  }

  friend Vector3D operator& (const Vector3D &a, const Vector3D &b){
    //cross product
    Vector3D r = Vector3D(a.y*b.z - a.z*b.y, a.z*b.z - a.x*b.z, a.x*b.y - a.y*b.z);
    return r;
  }

  friend T operator| (const Vector3D &a, const Vector3D &b){
    //dot product
    return (T) (a.x*b.x + a.y*b.y + a.z*b.z);
  }

  //double* ToDoubleArray() { double* a = new double[3]; a[0] = x; a[1] = y; a[2] = z; return a; }
  //void FromDoubleArray(double *a) { x = a[0]; y = a[1]; z = a[2]; }

  T norm() { return sqrt(x*x + y*y + z*z); }

  Vector3D(float *f){ x = (T) f[0]; y = (T) f[1]; z = (T) f[2]; }
  Vector3D(double *d) { x = (T) d[0]; y = (T) d[1]; z = (T) d[2]; }
  Vector3D(long double *d) { x = (T) d[0]; y = (T) d[1]; z = (T) d[2]; }
  Vector3D(int *i) { x = (T) i[0]; y = (T) i[1]; z = (T) i[2]; }

  Vector3D(float f0, float f1, float f2) { x = (T)f0; y = (T)f1; z = (T)f2; }
  Vector3D(double f0, double  f1, double f2){ x = (T)f0; y = (T)f1; z = (T)f2; }
  Vector3D(long double f0, long double  f1, long double f2){ x = (T)f0; y = (T)f1; z = (T)f2; }
  Vector3D(int i0, int i1, int i2) { x = (T)i0; y = (T)i1; z = (T)i2; }

  Vector3D(void) { x = 0; y = 0; z = 0; }

  void CopyInto(Vector3D<T> dest) {
    dest.x = this->x; dest.y = this->y; dest.z = this->z;
  }
  void CopyInto(Vector3D<T> *dest) {
    dest->x = this->x; dest->y = this->y; dest->z = this->z;
  }
  ~Vector3D(void) {}
};
}
}

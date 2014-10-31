#pragma once

namespace PerseusLib
{
namespace Primitives
{
template <typename T>
class Vector4D
{
public:
  T x, y, z, w;

  const Vector4D& operator+= (Vector4D &p){
    x += p.x; y += p.y; z += p.z; z += p.w;
    return *this;
  }

  const Vector4D& operator-= (Vector4D &p){
    x -= p.x; y -= p.y; z -= p.z; z += p.w;
    return *this;
  }

  Vector4D operator* (T f) const{
    Vector4D r(x * f, y * f, z * f, w * f);
    return r;
  }

  Vector4D& operator*= (T f){
    x *= f; y *= f; z *= f; z *= w;
    return *this;
  }

  friend Vector4D operator+ (const Vector4D &lp, const Vector4D &rp){
    Vector4D r = Vector4D(lp.x + rp.x, lp.y + rp.y, lp.z + rp.z, lp.w + rp.w);
    return r;
  }

  friend Vector4D operator- (const Vector4D &lp, const Vector4D &rp){
    Vector4D r = Vector4D(lp.x - rp.x, lp.y - rp.y, lp.z - rp.z, lp.w + rp.w);
    return r;
  }

  //friend Vector4D operator& (const Vector4D &a, const Vector4D &b){
  //	//cross product
  //	Vector4D r = Vector4D(
  //		a.y*(b.z*cw - cz*b.w) - a.z*(b.y*cw - cy*b.w) + a.w*(b.y*cz - cy*b.z),
  //		-a.x*(b.z*cw - cz*b.w) + a.z*(b.x*cw - cx*b.w) - a.w*(b.x*cz - cx*b.z),
  //		a.x*(b.y*cw - cy*b.w) - a.y*(b.x*cw - cx*b.w) + a.w*(b.x*cy - cx*b.y),
  //		-a.x*(b.y*cz - cy*b.z) + a.y*(b.x*cz - cx*b.z) - a.z*(b.x*cy - cx*b.y)
  //		);
  //	return r;
  //}

  friend T operator| (const Vector4D &a, const Vector4D &b){
    //dot product
    return (T) (a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w);
  }

  friend bool operator== (const Vector4D &lp, const Vector4D &rp){
    return ((lp.x == rp.x) && (lp.y == rp.y) && (lp.z == rp.z) && (lp.w == rp.w));
  }

  Vector4D(float *f){ x = (T) f[0]; y = (T) f[1]; z = (T) f[2]; w = (T) f[3]; }
  Vector4D(double *d) { x = (T) d[0]; y = (T) d[1]; z = (T) d[2]; w = (T) d[3]; }
  Vector4D(long double *d) { x = (T) d[0]; y = (T) d[1]; z = (T) d[2]; w = (T) d[3]; }
  Vector4D(int *i) { x = (T) i[0]; y = (T) i[1]; z = (T) i[2]; w = (T) i[3]; }

  Vector4D(float v0, float v1, float v2, float v3) { x = (T)v0; y = (T)v1; z = (T)v2; w = (T)v3; }
  Vector4D(double v0, double  v1, double v2, double v3){ x = (T) v0; y = (T) v1; z = (T) v2; w = (T) v3; }
  Vector4D(long double v0, long double  v1, long double v2, long double v3)
  { x = (T) v0; y = (T) v1; z = (T) v2; w = (T) v3; }
  Vector4D(int v0, int v1, int v2, int v3) { x = (T) v0; y = (T) v1; z = (T) v2; w = (T) v3; }

  Vector4D(void) { x = 0; y = 0; z = 0; w = 0; }

  ~Vector4D(void) {}
};
}
}

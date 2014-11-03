#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Primitives/Vector3D.h>
#include <PerseusLib/Primitives/Vector4D.h>

#include <PerseusLib/Renderer/Primitives/Quaternion.h>

using namespace PerseusLib::Primitives;

#include <stdio.h>

#include <cmath>

#ifndef SIGNUMF
#define SIGNUMF(val) ( (val < 0.0f) ? -1.0f : 1.0f )
#endif

#ifndef ISIGNUMF
#define ISIGNUMF(val) ( (val < 0.0f) ? 1.0f : -1.0f )
#endif

namespace Renderer
{
namespace Primitives
{
class Quaternion
{
  void matrixToOpenGLMatrix(VFLOAT m[][4], VFLOAT* M);
  void toOpenGLMatrix(VFLOAT m[][4]);
  int nrOfRotationVariables;

public:
  VECTOR4DA vector4d;

  void Set(VFLOAT quatX, VFLOAT quatY, VFLOAT quatZ, VFLOAT quatW) {
    this->vector4d.x = quatX;
    this->vector4d.y = quatY;
    this->vector4d.z = quatZ;
    this->vector4d.w = quatW;
  }

  void Set(VFLOAT* quaternion) {  vector4d = VECTOR4DA(quaternion); }

  void Set(Quaternion* qrotation) {
    qrotation->CopyInto(this);
  }

  Quaternion(VFLOAT x, VFLOAT y, VFLOAT z, VFLOAT w) {
    vector4d = VECTOR4DA(x, y, z , w);
    nrOfRotationVariables = 4;
  }
  Quaternion(VFLOAT rotationX, VFLOAT rotationY, VFLOAT rotationZ) { FromEuler(rotationX, rotationY, rotationZ); nrOfRotationVariables = 4; }
  Quaternion(VFLOAT* rotation) { FromEuler(rotation); nrOfRotationVariables = 4; }

  void FromEuler(VFLOAT rotationX, VFLOAT rotationY, VFLOAT rotationZ);
  void FromEuler(VFLOAT* rotation) { FromEuler(rotation[0], rotation[1], rotation[2]); }
  void FromMatrix(VFLOAT* rotMatrix);

  void ToOpenGLMatrix(VFLOAT*);

  Quaternion* Clone() { Quaternion* q = new Quaternion(vector4d.x, vector4d.y, vector4d.z, vector4d.w); return q; }
  void CopyInto(Quaternion *q) { q->vector4d = VECTOR4DA(this->vector4d.x, this->vector4d.y, this->vector4d.z, this->vector4d.w); }

  void SetFromMatrix(VFLOAT *matrix) { FromMatrix(matrix); }

  void GetMatrix(VFLOAT *output) { this->ToOpenGLMatrix(output); }
  void Get(VFLOAT* parameters) {
    //        printf("[Quaternion/Get] \n");

    parameters[0] = vector4d.x;
    parameters[1] = vector4d.y;
    parameters[2] = vector4d.z;
    parameters[3] = vector4d.w;
  }
  void GetDerivatives(VFLOAT* derivatives, VFLOAT* xUnprojected, VFLOAT* xSource, VFLOAT* projectionParams, VFLOAT* otherInfo);

  void Add(Quaternion* qrotation) {
    Quaternion q;
    q = (*((Quaternion*)qrotation)) * (Quaternion)*this;
    q.CopyInto(this);
  }

  void SumOf(Renderer::Primitives::Quaternion* rsrc1, Renderer::Primitives::Quaternion* rsrc2){
    Quaternion *src1 = (Quaternion*)rsrc1;
    Quaternion *src2 = (Quaternion*)rsrc2;
    Quaternion sum;

    sum = *src1 * *src2;

    //sum.vector4d.w = src1->vector4d.w*src1->vector4d.w - src1->vector4d.x*src1->vector4d.x -
    //	src1->vector4d.y*src1->vector4d.y - src1->vector4d.z*src1->vector4d.z;
    //sum.vector4d.x = src1->vector4d.w*src1->vector4d.x +  src1->vector4d.x*src1->vector4d.w +
    //	src1->vector4d.y*src1->vector4d.z - src1->vector4d.z*src1->vector4d.y;
    //sum.vector4d.y = src1->vector4d.w*src1->vector4d.y - src1->vector4d.x*src1->vector4d.z +
    //	src1->vector4d.y*src1->vector4d.w + src1->vector4d.z*src1->vector4d.x;
    //sum.vector4d.z = src1->vector4d.w*src1->vector4d.z + src1->vector4d.x*src1->vector4d.y -
    //	src1->vector4d.y*src1->vector4d.x + src1->vector4d.z*src1->vector4d.w;

    sum.CopyInto(this);
  }

  void AddPost(Quaternion* qrotation) {
    Quaternion q;
    q = (Quaternion)*this * (*((Quaternion*)qrotation));
    q.CopyInto(this);
  }

  void SetFromEuler(VFLOAT rotationX, VFLOAT rotationY, VFLOAT rotationZ)
  {
    FromEuler(rotationX, rotationY, rotationZ);
  }

  void Normalize()
  {
    VFLOAT norm = 1 / sqrtf(
          vector4d.x * vector4d.x + vector4d.y * vector4d.y +
          vector4d.z * vector4d.z + vector4d.w * vector4d.w
          );

    vector4d.x *= norm; vector4d.y *= norm;
    vector4d.z *= norm; vector4d.w *= norm;
  }

  friend Quaternion operator* (const Quaternion &q1, const Quaternion &q2){
    Quaternion prod;

    prod.vector4d.x =  q1.vector4d.x * q2.vector4d.w + q1.vector4d.y * q2.vector4d.z
        - q1.vector4d.z * q2.vector4d.y + q1.vector4d.w * q2.vector4d.x;
    prod.vector4d.y = -q1.vector4d.x * q2.vector4d.z + q1.vector4d.y * q2.vector4d.w
        + q1.vector4d.z * q2.vector4d.x + q1.vector4d.w * q2.vector4d.y;
    prod.vector4d.z =  q1.vector4d.x * q2.vector4d.y - q1.vector4d.y * q2.vector4d.x
        + q1.vector4d.z * q2.vector4d.w + q1.vector4d.w * q2.vector4d.z;
    prod.vector4d.w = -q1.vector4d.x * q2.vector4d.x - q1.vector4d.y * q2.vector4d.y
        - q1.vector4d.z * q2.vector4d.z + q1.vector4d.w * q2.vector4d.w;

    prod.Normalize();
    prod.vector4d.x = q1.vector4d.w*q2.vector4d.x + q1.vector4d.x*q2.vector4d.w
        + q1.vector4d.y*q2.vector4d.z - q1.vector4d.z*q2.vector4d.y;
    prod.vector4d.y = q1.vector4d.w*q2.vector4d.y + q1.vector4d.y*q2.vector4d.w
        + q1.vector4d.z*q2.vector4d.x - q1.vector4d.x*q2.vector4d.z;
    prod.vector4d.z = q1.vector4d.w*q2.vector4d.z + q1.vector4d.z*q2.vector4d.w
        + q1.vector4d.x*q2.vector4d.y - q1.vector4d.y*q2.vector4d.x;
    prod.vector4d.w = q1.vector4d.w*q2.vector4d.w - q1.vector4d.x*q2.vector4d.x
        - q1.vector4d.y*q2.vector4d.y - q1.vector4d.z*q2.vector4d.z;


    return prod;
  }

  void ToEuler(VECTOR3DA *euler)
  {
    float sqw = this->vector4d.w*this->vector4d.w;
    float sqx = this->vector4d.x*this->vector4d.x;
    float sqy = this->vector4d.y*this->vector4d.y;
    float sqz = this->vector4d.z*this->vector4d.z;
    float unit = sqx + sqy + sqz + sqw;
    float test = this->vector4d.x*this->vector4d.y + this->vector4d.z*this->vector4d.w;

    // singularity at north pole
    if (test > 0.499*unit)
    {
      euler->y = 2 * atan2f(this->vector4d.x,this->vector4d.w);
      euler->z = (float)PI;
      euler->x = 0;

      euler->x /= (float)DEGTORAD; euler->y /= (float)DEGTORAD; euler->z /= (float)DEGTORAD;

      return;
    }

    // singularity at south pole
    if (test < -0.499*unit)
    {
      euler->y = -2 * atan2f(this->vector4d.x,this->vector4d.w);
      euler->z = -(float)PI;
      euler->x = 0;

      euler->x /= (float)DEGTORAD; euler->y /= (float)DEGTORAD; euler->z /= (float)DEGTORAD;

      return;
    }

    euler->y = atan2f(2*this->vector4d.y*this->vector4d.w-2*this->vector4d.x*this->vector4d.z , sqx - sqy - sqz + sqw);
    euler->z = asinf(2*test/unit);
    euler->x = atan2f(2*this->vector4d.x*this->vector4d.w-2*this->vector4d.y*this->vector4d.z , -sqx + sqy - sqz + sqw);

    euler->x /= (float)DEGTORAD; euler->y /= (float)DEGTORAD; euler->z /= (float)DEGTORAD;
  }

  void Invert()
  {
    this->vector4d.x *= -1;
    this->vector4d.y *= -1;
    this->vector4d.z *= -1;
  }

  void FromPointAndReference(float refX, float refY, float refZ, float pointX, float pointY, float pointZ)
  {
    float axisx, axisy, axisz;

    axisx = pointX*refZ - pointZ*refX;
    axisy = pointY*refZ - pointZ*refY;
    axisz = pointY*refX - pointX*refY;

    this->vector4d.x = axisx;
    this->vector4d.y = axisy;
    this->vector4d.z = axisz;
    this->vector4d.w = 1.0f + pointX*refX + pointY*refY + pointZ*refZ;

    this->Normalize();
  }

  void FromPointAndReference(float refX, float refY, float refZ, float pointX, float pointY, float pointZ, int type)
  {
    float axisx, axisy, axisz;
    float axisxCand[3], axisyCand[3];// axiszCand[6];

    if (sqrtf(pointX * pointX + pointY * pointY + pointZ * pointZ) != 0 &&
        sqrtf(refX * refX + refY * refY + refZ * refZ) != 0)
    {
      pointX = pointX / sqrtf(pointX * pointX + pointY * pointY + pointZ * pointZ);
      pointY = pointY / sqrtf(pointX * pointX + pointY * pointY + pointZ * pointZ);
      pointZ = pointZ / sqrtf(pointX * pointX + pointY * pointY + pointZ * pointZ);

      refX = refX / sqrtf(refX * refX + refY * refY + refZ * refZ);
      refY = refY / sqrtf(refX * refX + refY * refY + refZ * refZ);
      refZ = refZ / sqrtf(refX * refX + refY * refY + refZ * refZ);

      axisx = 0.000001f;
      axisy = 0.000001f;
      axisz = 0.000001f;

      axisxCand[0] = sqrtf(pointZ * pointZ + pointY * pointY) * refX * SIGNUMF(pointZ) - pointX * sqrtf(refZ * refZ + refY * refY) * SIGNUMF(refZ);
      axisxCand[1] = sqrtf(pointZ * pointZ + pointY * pointY) * refX - pointX * sqrtf(refZ * refZ + refY * refY);
      axisxCand[2] = pointZ * refX - pointX * refZ;
      axisx = axisxCand[0];
      for (int i=1; i<3; i++)
      { if (int(1000 * std::abs(axisx)) != int(1000 * std::abs(axisxCand[i]))) axisx = (std::abs(axisx) < std::abs(axisxCand[i])) ? axisx : axisxCand[i]; }

      axisyCand[0] = sqrtf(pointZ * pointZ + pointX * pointX) * refY * SIGNUMF(pointZ) - pointY * sqrtf(refZ * refZ + refX * refX) * SIGNUMF(refZ);
      axisyCand[1] = sqrtf(pointZ * pointZ + pointX * pointX) * refY - pointY * sqrtf(refZ * refZ + refX * refX);
      axisyCand[2] = pointZ * refY - pointY * refZ;
      axisy = axisyCand[0];
      for (int i=1; i<3; i++)
      { if (int(1000 * std::abs(axisy)) != int(1000 * std::abs(axisyCand[i]))) axisy = (std::abs(axisy) < std::abs(axisyCand[i])) ? axisy : axisyCand[i]; }

      //axiszCand[0] = pointX * refY - pointY * refX;
      //axiszCand[1] = pointY * refX - pointX * refY;
      //axiszCand[1] = sqrtf(pointY * pointY + pointX * pointX) * refY * SIGNUMF(pointX) - pointY * sqrtf(refY * refY + refX * refX) * SIGNUMF(refX);
      //axiszCand[2] = sqrtf(pointY * pointY + pointX * pointX) * refX * SIGNUMF(pointY) - pointX * sqrtf(refY * refY + refX * refX) * SIGNUMF(refY);
      //axiszCand[3] = sqrtf(pointY * pointY + pointX * pointX) * refY - pointY * sqrtf(refY * refY + refX * refX);
      //axiszCand[4] = sqrtf(pointY * pointY + pointX * pointX) * refX - pointX * sqrtf(refY * refY + refX * refX);

      //axisz = axiszCand[0];
      //for (int i=1; i<2; i++)
      //{ if (int(1000 * abs(axisz)) != int(1000 * abs(axiszCand[i]))) axisz = (abs(axisz) < abs(axiszCand[i])) ? axisz : axiszCand[i]; }

      switch (type)
      {
      case 0:
        this->vector4d.x = axisx; this->vector4d.y = 0; this->vector4d.z = 0;
        this->vector4d.w = 1.0f + pointY*refY + pointZ*refZ + pointX*refX;
        break;
      case 1:
        this->vector4d.x = 0; this->vector4d.y = axisy; this->vector4d.z = 0;
        this->vector4d.w = 1.0f + pointX*refX + pointY*refY + pointZ*refZ;
        break;
      case 2:
        this->vector4d.x = 0; this->vector4d.y = 0; this->vector4d.z = axisz;
        this->vector4d.w = 1.0f + pointX*refX + pointY*refY + pointZ*refZ;
        break;
      case 3:
        this->vector4d.x = axisy; this->vector4d.y = axisx; this->vector4d.z = 0;
        this->vector4d.w = 1.0f + pointX*refX + pointY*refY + pointZ*refZ;
        break;
      }

      this->Normalize();
    }
    else
    {
      this->vector4d.x = 0;
      this->vector4d.y = 0;
      this->vector4d.z = 0;
      this->vector4d.w = 1;
    }
  }

  Quaternion(void) { nrOfRotationVariables = 4; this->vector4d.x = 0; this->vector4d.y = 0; this->vector4d.z = 0; this->vector4d.w = 1; }
  ~Quaternion(void) {}
};
}
}

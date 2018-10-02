#include "quaternion.h"

using namespace cv;
using namespace std;

// Note: These two classes are based on the boxplus method described in C. Hertzberg, R. Wagner, U. Frese, and L. Schroder,
// "Integrating generic sensor fusion algorithms with sound state representations through encapsulation of manifolds,”
// Information Fusion, vol. 14, no. 1, pp. 57–77, 2011.

// Class vect (vectview class in the paper)
vect::vect(double p_, double q_, double r_)
{
    p = p_;
    q = q_;
    r = r_;
}

vect vect::operator*(const double& scale)
{
    return vect(p*scale, q*scale, r*scale);
}

double vect::length()
{
    return sqrt(p*p + q*q + r*r);
}

// Quaternion class
double sinc(double x)
{
    // Taylor series expansion of sin is:           x - x^3/3! + x^5/5! - ...
    // Thus the Taylor series expansion of sinc is: 1 - x^2/3! + x^4/5! - ...
    // Double precision is approximately 16 digits. Since the largest term is x^2,
    // we will experience numerical underflow if |x| < 1e-8.
    // Of course, floating point arithmetic can handle much smaller values than this (as small as 1e-308).
    // I haven't seen any problems with small numbers so far, so we could just check for division by zero,
    // but this solution is guarenteed not to have problems.
    if (fabs(x) < 1e-8)
        return 1;
    else
        return sin(x) / x;
}

quaternion::quaternion(double s_, double x_, double y_, double z_)
{
    s = s_;
    x = x_;
    y = y_;
    z = z_;
}

quaternion::quaternion(vect v)
{
    double theta = v.length();
    double scale = sinc(theta / 2) / 2;
    s = cos(theta / 2);
    x = scale*v.p;
    y = scale*v.q;
    z = scale*v.r;
}

vect quaternion::toVec()
{
    double theta = acos(s)*2;
    double scale = sinc(theta / 2) / 2;
    return vect(x/scale, y/scale, z/scale);
}

Mat quaternion::toRotationMatrix()
{
    return (Mat_<float>(3,3) <<
        1-2*(y*y + z*z),   2*(x*y - z*s),   2*(x*z + y*s),
          2*(x*y + z*s), 1-2*(x*x + z*z),   2*(y*z - x*s),
          2*(x*z - y*s),   2*(y*z + x*s), 1-2*(x*x + y*y));
}

quaternion quaternion::operator*(const quaternion& q2)
{
    return quaternion(
        s*q2.s - x*q2.x - y*q2.y - z*q2.z,
        x*q2.s + s*q2.x - z*q2.y + y*q2.z,
        y*q2.s + z*q2.x + s*q2.y - x*q2.z,
        z*q2.s - y*q2.x + x*q2.y + s*q2.z
    );
}

quaternion quaternion::operator/(const double& scale)
{
    return quaternion(s/scale, x/scale, y/scale, z/scale);
}

quaternion quaternion::conj()
{
    return quaternion(s, -x, -y, -z);
}

double quaternion::length()
{
    return sqrt(s*s + x*x + y*y + z*z);
}

quaternion quaternion::inv()
{
    return conj() / length();
}

void quaternion::mult(quaternion q2)
{
    quaternion q3 = *this * q2;
    s = q3.s;
    x = q3.x;
    y = q3.y;
    z = q3.z;
}

// y = x [+] delta
void quaternion::boxplus(vect delta, double scale) //scale = 1
{
    mult(quaternion(delta*scale));
}

// delta = y [-] x
vect quaternion::boxminus(quaternion x)
{
    quaternion q3 = x.inv() * *this;
    return q3.toVec();
}
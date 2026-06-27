#include "kml_math_test.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "km.h"


int kml_math_sin_01()
{
    double pi = acos(-1); 
    // typical usage 
    double a = pi/6, b = 1.0, c = -3*pi/4, d = pi/3; 
    // special handling 
    double e = INFINITY, f = -INFINITY, g = NAN; 
    sin(a);

    return 1; // 成功
}
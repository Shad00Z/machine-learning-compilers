#ifndef MINI_JIT_CONSTANTS_H
#define MINI_JIT_CONSTANTS_H

const float FLOAT_ERROR_MARGIN = 0.001f;

const float sig_table[33] = {
    0.000335f, 0.000553f, 0.000911f, 0.001503f, 0.002473f, 0.004070f, 0.006693f,
    0.011109f, 0.017986f, 0.029312f, 0.047426f, 0.075858f, 0.119203f, 0.182426f,
    0.268941f, 0.377541f, 0.500000f, 0.622459f, 0.731059f, 0.817574f, 0.880797f,
    0.924142f, 0.952574f, 0.970688f, 0.982014f, 0.988891f, 0.993307f, 0.995930f,
    0.997527f, 0.998497f, 0.999089f, 0.999447f, 0.999665f
};

const float sig_taylor_values[16] = {
          0.5f,       0.5f,       0.5f,       0.5f,
         0.25f,      0.25f,      0.25f,      0.25f, 
    -0.020833f, -0.020833f, -0.020833f, -0.020833f,
     0.002083f,  0.002083f,  0.002083f,  0.002083f
};

#endif // MINI_JIT_CONSTANTS_H
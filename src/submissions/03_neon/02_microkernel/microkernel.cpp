#include <cstdint>
#include <iostream>

extern "C"
{
    /**
     * @param a pointer to column-major matrix A.
     * @param b pointer to column-major matrix B.
     * @param c pointer to column-major matrix C.
     * @param lda leading dimension of A.
     * @param ldb leading dimension of B.
     * @param ldc leading dimension of C.
     **/
    void v1_matmul_16_6_1( float const *a,
                           float const *b,
                           float *c,
                           int64_t lda,
                           int64_t ldb,
                           int64_t ldc );

    /**
     * @param a pointer to column-major matrix A.
     * @param b pointer to column-major matrix B.
     * @param c pointer to column-major matrix C.
     * @param lda leading dimension of A.
     * @param ldb leading dimension of B.
     * @param ldc leading dimension of C.
     **/
    void v2_matmul_16_6_1( float const *a,
                           float const *b,
                           float *c,
                           int64_t lda,
                           int64_t ldb,
                           int64_t ldc );

    /**
     * @param a pointer to column-major matrix A.
     * @param b pointer to column-major matrix B.
     * @param c pointer to column-major matrix C.
     * @param lda leading dimension of A.
     * @param ldb leading dimension of B.
     * @param ldc leading dimension of C.
     **/
    void v3_matmul_16_6_1( float const *a,
                           float const *b,
                           float *c,
                           int64_t lda,
                           int64_t ldb,
                           int64_t ldc );
}

int main()
{
    const int M = 16;
    const int N = 6;
    const int K = 1;

    float A[M * K];
    float B[K * N];
    float C[M * N] = {0.0f};

    // Initialize matrices
    for ( int i = 0; i < M * K; ++i )
    {
        A[i] = static_cast<float>( i );
    }
    for ( int j = 0; j < K * N; ++j )
    {
        B[j] = static_cast<float>( j );
    }

    // Print matrix A in column-major order
    std::cout << "Matrix A (Column-Major Order):" << std::endl;
    for ( int row = 0; row < M; ++row )
    {
        for ( int col = 0; col < K; ++col )
        {
            std::cout << A[col * M + row] << " ";
        }
        std::cout << std::endl;
    }

    // Print matrix B in column-major order
    std::cout << "Matrix B (Column-Major Order):" << std::endl;
    for ( int row = 0; row < K; ++row )
    {
        for ( int col = 0; col < N; ++col )
        {
            std::cout << B[col * K + row] << " ";
        }
        std::cout << std::endl;
    }

    v1_matmul_16_6_1( A,
                      B,
                      C,
                      16,
                      1,
                      16 );

    // Print matrix C in column-major order
    std::cout << "Matrix C (Column-Major Order):" << std::endl;
    for ( int row = 0; row < N; ++row )
    {
        for ( int col = 0; col < M; ++col )
        {
            std::cout << C[col * N + row] << " ";
        }
        std::cout << std::endl;
    }
    C[M * N] = {0.0f};

    v2_matmul_16_6_1( A,
                      B,
                      C,
                      16,
                      1,
                      16 );

    // Print matrix C in column-major order
    std::cout << "Matrix C (Column-Major Order):" << std::endl;
    for ( int row = 0; row < N; ++row )
    {
        for ( int col = 0; col < M; ++col )
        {
            std::cout << C[col * N + row] << " ";
        }
        std::cout << std::endl;
    }
    C[M * N] = {0.0f};

    v3_matmul_16_6_1( A,
                      B,
                      C,
                      16,
                      1,
                      16 );

    // Print matrix C in column-major order
    std::cout << "Matrix C (Column-Major Order):" << std::endl;
    for ( int row = 0; row < N; ++row )
    {
        for ( int col = 0; col < M; ++col )
        {
            std::cout << C[col * N + row] << " ";
        }
        std::cout << std::endl;
    }
    C[M * N] = {0.0f};

    return 0;
}
#include <stdio.h>
#include <stdlib.h>


float* matmul(float* a, float* b, int m, int n, int p){
    float* c = (float*)malloc(sizeof(float)*m*p);

    for (int i = 0; i < m; i++){
        for (int j = 0; j < p; j++){
            float accumulate = 0;
            for (int k = 0; k < n; k++){
                accumulate += a[i*n+k] * b[k*p+j];
            }
            c[i*p+j] = accumulate;
        }
    }
    return c;
}

int main(){
    int m = 2;
    int n = 5;
    int p = 2;
    float* a = (float*)malloc(sizeof(float)*m*n);
    for (int i = 0; i < m*n; i++){
        a[i] = 2.0;
    }
    float* b = (float*)malloc(sizeof(float)*n*p);
    for (int i = 0; i < n*p; i++){
        b[i] = 2.0;
    }
    float* c = matmul(a, b, n, m, p);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < p; j++){
            printf("%f, ", c[i*n+j]);
        }
        printf("\n");
    }
    return 0;
}
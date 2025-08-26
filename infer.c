#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int all_close(float* a, float* b, int n){
	for (int i = 0; i < n; i++){
		if (a[i] != b[i]){
			return 0;
		}
	}
	return 1;
}

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

#include <stdlib.h>

float* transposed_matmul(float* a, float* b, int m, int n, int p) {
	// c: m x p
	float* c  = (float*)malloc((size_t)m * (size_t)p * sizeof(float));
	if (!c) return NULL;

	// bt: p x n  (transpose of b which is n x p)
	float* bt = (float*)malloc((size_t)p * (size_t)n * sizeof(float));
	if (!bt) {
		free(c);
		return NULL;
	}

	// Transpose b into bt so we can read rows of bt contiguously
	for (int j = 0; j < p; ++j) {
		for (int k = 0; k < n; ++k) {
			bt[j * n + k] = b[k * p + j];
		}
	}

	// Matmul using bt: C[i,j] = dot(A[i,:], BT[j,:])
	for (int i = 0; i < m; ++i) {
		const float* ai = &a[i * n];   // row i of A
		for (int j = 0; j < p; ++j) {
			const float* btj = &bt[j * n]; // row j of BT (i.e., col j of B)
			float sum = 0.0f;
			for (int k = 0; k < n; ++k) {
				sum += ai[k] * btj[k];
			}
			c[i * p + j] = sum;
		}
	}

	free(bt);
	return c;
}


int main(){
	int m = 1024;
	int n = 1024;
	int p = 1024;
	float* a = (float*)malloc(sizeof(float)*m*n);
	float* b = (float*)malloc(sizeof(float)*n*p);
	clock_t t0;
	clock_t t1;
	t0 = clock();
	float* c = transposed_matmul(a, b, m, n, p);
	float* d = transposed_matmul(a, b, m, n, p);
	t1 = clock();
	int closeness = all_close(c, d, m*p);
	printf("%d\n", closeness);
	printf("\n%f\n", ((double)t1-t0)/CLOCKS_PER_SEC);
	printf("%ld", CLOCKS_PER_SEC);
	free(a); free(b); free(c), free(d);
	return 0;
}
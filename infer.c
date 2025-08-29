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
		for (int k = 0; k < n; k++){
			for (int j = 0; j < p; j++){
				c[i*p+j] += a[i*n+k] * b[k*p+j];
			}
		}
	}
	return c;
}

int main(){
	int t = 1024;
	int n_embd = 512;

	float* c_attn = (float*)malloc(sizeof(float)*n_embd*n_embd*3);
	float* c_proj = (float*)malloc(sizeof(float)*n_embd*n_embd);

	float* x = (float*)malloc(sizeof(float)*t*n_embd);

	float* qkv = matmul(x, c_attn, t, n_embd, n_embd);
	float* q = &qkv[t*n_embd*0];
	float* k = &qkv[t*n_embd*1];
	float* v = &qkv[t*n_embd*2];
	return 0;
}
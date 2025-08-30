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

float* transpose(float* a, int m, int n){
	float* b = (float*)malloc(sizeof(float)*m*n);
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			b[j*m+i] = a[i*n+j];
		}
	}
	return b;
}

float* matmul(float* a, float* b, int m, int n, int p){
	float* c = (float*)malloc(sizeof(float)*m*p);
	for(int i = 0; i < m*p; i++){
		c[i] = 0.0;
	}
	for (int i = 0; i < m; i++){
		for (int k = 0; k < n; k++){
			for (int j = 0; j < p; j++){
				c[i*p+j] += a[i*n+k] * b[k*p+j];
			}
		}
	}
	return c;
}

float* mat_scalar_add(float* mat, float scalar, int m, int n){
	for(int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			mat[i*n+j] += scalar;
		}
	}
	return mat;
}

float* mat_scalar_mul(float* mat, float scalar, int m, int n){
	for(int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			mat[i*n+j] *= scalar;
		}
	}
	return mat;
}

int main(){	
	int t = 1024;
	int n_embd = 512;
	int n_heads = 8;
	int head_embd = 64;

	float* c_attn = (float*)malloc(sizeof(float)*n_embd*n_embd*3);
	for (int i = 0; i < n_embd*n_embd*3; i++){
		c_attn[i] = i % 256;
	}
	// float* c_proj = (float*)malloc(sizeof(float)*n_embd*n_embd);

	float* x = (float*)malloc(sizeof(float)*t*n_embd);
		for (int i = 0; i < t*n_embd; i++){
		x[i] = i % 256;
	}
	
	float* qkv = matmul(x, c_attn, t, n_embd, 3*n_embd);
	free(x); free(c_attn);
	float* q = (float*)malloc(sizeof(float)*t*n_embd);
	float* k = (float*)malloc(sizeof(float)*t*n_embd);
	float* v = (float*)malloc(sizeof(float)*t*n_embd);

	for (int i = 0; i < t; i++){
		for (int j = 0; j < n_embd; j++){
			q[i*n_embd+j] = qkv[i*3*n_embd + 0*n_embd+j];
			k[i*n_embd+j] = qkv[i*3*n_embd + 1*n_embd+j];
			v[i*n_embd+j] = qkv[i*3*n_embd + 2*n_embd+j];
		}
	}
	free(qkv);

	float* q_T = (float*)malloc(sizeof(float)*t*n_embd);
	float* k_T = (float*)malloc(sizeof(float)*t*n_embd);
	float* v_T = (float*)malloc(sizeof(float)*t*n_embd);

	for (int t_idx = 0; t_idx < t; t_idx++) {
		for (int h = 0; h < n_heads; h++) {
			for (int d = 0; d < head_embd; d++) {
				int flat_idx = t_idx * (n_heads * head_embd) + h * head_embd + d;
				int T_idx = h * (t * head_embd) + t_idx * head_embd + d;
				q_T[T_idx] = q[flat_idx];
				k_T[T_idx] = k[flat_idx];
				v_T[T_idx] = v[flat_idx];
			}
		}
	}
	free(q); free(k); free(v);
	float* result = (float*)malloc(sizeof(float)*n_heads*t*t);
	for (int k = 0; k < n_heads; k++){
		float* left = &q_T[k*(t*head_embd)];
		float* right = &k_T[k*(t*head_embd)];
		float* right_T = transpose(right, t, head_embd);
		float* temp = matmul(left, right_T, t, head_embd, t);
		for (int i = 0; i < t; i++){
			for (int j = 0; j < t; j++){
				result[k * (t*t) + i*t + j] = temp[i*t+j];
			}
		}
	}
	printf("%f", result[8432]);

	return 0;
}
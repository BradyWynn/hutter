#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

FILE *file;

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

float* mat_scalar_add(float* mat, float scalar, int n){
	for (int i = 0; i < n; i++){
		mat[i] += scalar;
	}
	return mat;
}

float* mat_scalar_mul(float* mat, float scalar, int n){
	for (int i = 0; i < n; i++){
		mat[i] *= scalar;
	}
	return mat;
}

float* scaled_dot_product_attention(float* q, float* k, float* v, int t, int n_embd, int head_embd){
	int n_heads = n_embd / head_embd;

	// q @ k_T
	float* result = (float*)malloc(sizeof(float)*n_heads*t*t);
	for (int h = 0; h < n_heads; h++){
		float* left = &q[h*(t*head_embd)];
		float* right = &k[h*(t*head_embd)];
		float* right_T = transpose(right, t, head_embd);
		float* temp = matmul(left, right_T, t, head_embd, t);
		for (int i = 0; i < t; i++){
			for (int j = 0; j < t; j++){
				result[h * (t*t) + i*t + j] = temp[i*t+j];
			}
		}
	}
	// divide d_k
	result = mat_scalar_mul(result, 1 / 8.0, n_heads*t*t);
	// softmax
	for (int h = 0; h < n_heads; h++){
		for (int i = 0; i < t; i++){
			float max_val = -10000;
			float exp_sum = 0;
			for (int j = 0; j < i+1; j++){
				if (result[h*t*t + i*t + j] > max_val){
					max_val = result[h*t*t + i*t + j];
				}
			}
			for (int j = 0; j < i+1; j++){
				result[h*t*t + i*t + j] = exp(result[h*t*t + i*t + j] - max_val);
				exp_sum += result[h*t*t + i*t + j];
			}
			for (int j = 0; j < i+1; j++){
				result[h*t*t + i*t + j] = result[h*t*t + i*t + j] / exp_sum;
			}
			for (int j = 0; j < t; j++){
				if (j > i){
					result[h*t*t + i*t + j] = 0.0;
				}
			}
		}
	}
	float* attn_out = (float*)malloc(sizeof(float)*n_heads*t*head_embd);
	for(int i = 0; i < n_heads*t*head_embd; i++){
		attn_out[i] = 0.0;
	}
	// v
	for (int h = 0; h < n_heads; h++){
		for (int i = 0; i < t; i++){
			for (int k = 0; k < t; k++){
				for (int j = 0; j < head_embd; j++){
					attn_out[h*t*head_embd + i*head_embd+j] += result[h*t*t + i*t+k] * v[h*t*head_embd + k*head_embd+j];
				}
			}
		}
	}
	return attn_out;
}

int main(){
	char dir[] = "model/transformer.h.0.attn.c_attn.weight.npy";
	file = fopen(dir, "r");

	char header[128];
	size_t file_read = fread(header, sizeof(char), 128, file);

	float c_attn_T[1536*512];
	file_read = fread(c_attn_T, sizeof(float), 1536*512, file);

	float* c_attn = transpose(c_attn_T, 1536, 512);

	int t = 1024;
	int n_embd = 512;
	int head_embd = 64;
	int n_heads = n_embd / head_embd;

	float* x = (float*)malloc(sizeof(float)*t*n_embd);
	for (int i = 0; i < t*n_embd; i++){
		x[i] = 0.01;
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

	// norm
	float* q_accumulate = (float*)malloc(sizeof(float)*t*n_heads);
	for (int i = 0; i < head_embd; i++){
		q_accumulate[i] = 0.0;
	}
	for (int i = 0; i < t; i++){
		for (int j = 0; j < n_heads; j++){
			for (int k = 0; k < head_embd; k++){
				q_accumulate[i*n_heads+j] += pow(q[i*n_heads*head_embd+j*head_embd+k], 2.0);
			}
		}
	}
	float* k_accumulate = (float*)malloc(sizeof(float)*t*n_heads);
		for (int i = 0; i < head_embd; i++){
		k_accumulate[i] = 0.0;
	}
	for (int i = 0; i < t; i++){
		for (int j = 0; j < n_heads; j++){
			for (int r = 0; r < head_embd; r++){
				k_accumulate[i*n_heads+j] += pow(k[i*n_heads*head_embd+j*head_embd+r], 2.0);
			}
		}
	}
	for (int i = 0; i < t*n_heads; i++){
		q_accumulate[i] = sqrt(q_accumulate[i] / head_embd + 1e-8);
		k_accumulate[i] = sqrt(k_accumulate[i] / head_embd + 1e-8);
	}
	for (int i = 0; i < t; i++){
		for (int j = 0; j < n_heads; j++){
			float q_denom = q_accumulate[i*n_heads+j];
			float k_denom = k_accumulate[i*n_heads+j];
			for (int r = 0; r < head_embd; r++){
				q[i*n_heads*head_embd + j*head_embd + r] /= q_denom;
				k[i*n_heads*head_embd + j*head_embd + r] /= k_denom;	
			}
		}
	}
	for (int i = 0; i < 10; i++){
		printf("%f, ", q[i]);
	}
	// (t, n_embd) --> (t, n_heads, head_embd) --> (n_heads, t, head_embd)

	// float* q_T = (float*)malloc(sizeof(float)*t*n_embd);
	// float* k_T = (float*)malloc(sizeof(float)*t*n_embd);
	// float* v_T = (float*)malloc(sizeof(float)*t*n_embd);

	// // (n_heads, t, head_embd)
	// for (int t_idx = 0; t_idx < t; t_idx++) {
	// 	for (int h = 0; h < n_heads; h++) {
	// 		for (int d = 0; d < head_embd; d++) {
	// 			int flat_idx = t_idx * (n_heads * head_embd) + h * head_embd + d;
	// 			int T_idx = h * (t * head_embd) + t_idx * head_embd + d;
	// 			q_T[T_idx] = q[flat_idx];
	// 			k_T[T_idx] = k[flat_idx];
	// 			v_T[T_idx] = v[flat_idx];
	// 		}
	// 	}
	// }
	// free(q); free(k); free(v);

	// float* attn_out = scaled_dot_product_attention(q_T, k_T, v_T, t, n_embd, head_embd);

	// float sum = 0;
	// for (int i = 0; i < n_heads*t*head_embd; i++){
	// 	sum += attn_out[i];
	// }
	// printf("%f", sum);

	return 0;
}
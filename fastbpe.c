#include <stdio.h>
#include <stdlib.h>
#include <math.h>

FILE *data;
FILE *file;

typedef struct {
	int first;
	int second;
	int count;
} Bigram;

Bigram get_bigram(short* arr, int size){
	int counts[512][512] = {0};
	Bigram bigram;

	for(int i = 0; i < (size-1); i++){
		counts[arr[i]][arr[i+1]] += 1;
	}

	int highest_value = 0;
	for (int i = 0; i < 512; i++){
		for (int j = 0; j < 512; j++){
			if (counts[i][j] > highest_value){
				highest_value = counts[i][j];
				bigram.first = i;
				bigram.second = j;
				bigram.count = counts[i][j];
			}
		}
	}
	return bigram;
}

int replace_bigram(short* arr, Bigram bigram, int size, int new_token){
	int count = 0;
	for (int i = 0; i < size - 1; i++){
		if (arr[i] == bigram.first && arr[i+1] == bigram.second){
			arr[i] = new_token;
			arr[i+1] = 0;
			i += 1;
			count += 1;
		}
	}
	return count;
}

void remove_empty(short* arr, int size, int new_size){
	short* write_to = (short*)malloc(new_size * sizeof(short));
	int count = 0;
	for (int i = 0; i < size; i++){
		if (arr[i] != 0){
			write_to[count] = arr[i];
			count += 1;
		}
	}
	for (int i = 0; i < new_size; i++){
		arr[i] = write_to[i];
	}
	free(write_to);
}

int main(){
	int arr_size = pow(10, 9);

	char dir[] = "enwik9";
	char out[] = "tokenized_enwik9";
	data = fopen(dir, "r");

	__uint8_t* arr = (__uint8_t*)malloc(arr_size * sizeof(__uint8_t));
	fread(arr, sizeof(__uint8_t), arr_size, data);
	fclose(data);

	short* int16_arr = (short*)malloc(arr_size * sizeof(short));
	for (int i = 0; i < arr_size; i++){
		int16_arr[i] = (short)arr[i];
	}
	free(arr);

	for (int i = 0; i < 100; i++){
		Bigram bigram = get_bigram(int16_arr, arr_size);
		int count = replace_bigram(int16_arr, bigram, arr_size, 256+i);
		remove_empty(int16_arr, arr_size, (arr_size - count));
		arr_size = (arr_size - count);
		printf("%d | %d\n", arr_size, i);
	}
	
	file = fopen(out, "wb");
	fwrite(int16_arr, sizeof(short), arr_size, file);
	fclose(file);

	return 0;
}
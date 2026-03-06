#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

FILE* data;
FILE* file;

typedef struct {
    int first;
    int second;
    int count;
} Bigram;

int replace_and_compact(uint16_t* arr, int size, Bigram bigram, int new_token){
    int write_idx = 0;
    int i = 0;
    int replaced = 0;

    while (i < size){
        if (i < size - 1 && arr[i] == bigram.first && arr[i+1] == bigram.second){
            arr[write_idx++] = new_token;
            i += 2;
            replaced++;
        } else{
            arr[write_idx++] = arr[i++];
        }
    }
    return size - write_idx;
}

int main(){
	int arr_size = pow(10, 9);

    int num_tokens = 60000;

	char dir[] = "enwik9";
	char out[] = "60k_enwik9";
	data = fopen(dir, "r");
    FILE* first_file = fopen("first", "r");
    FILE* second_file = fopen("second", "r");

    int* first = (int*)malloc(num_tokens * sizeof(int));
    int* second = (int*)malloc(num_tokens * sizeof(int));

    fread(first, sizeof(int), num_tokens, first_file);
    fread(second, sizeof(int), num_tokens, second_file);

    fclose(first_file);
    fclose(second_file);

	__uint8_t* arr = (__uint8_t*)malloc(arr_size * sizeof(__uint8_t));
	fread(arr, sizeof(__uint8_t), arr_size, data);
	fclose(data);

	uint16_t* int16_arr = (uint16_t*)malloc(arr_size * sizeof(uint16_t));
	for (int i = 0; i < arr_size; i++){
		int16_arr[i] = (uint16_t)arr[i];
	}
	free(arr);

    for (int i = 0; i < num_tokens; i++){
        Bigram bigram = {first[i], second[i]};
        int removed = replace_and_compact(int16_arr, arr_size, bigram, 256 + i);
        arr_size -= removed;
        printf("%f%% | %d | %d\n", (float)i/num_tokens * 100, 1+i, arr_size);
    }
    
	file = fopen(out, "wb");
	fwrite(int16_arr, sizeof(uint16_t), arr_size, file);
	fclose(file);

	return 0;
}
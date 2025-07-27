#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TABLE_SIZE 2000000

FILE* data;
FILE* file;

typedef struct {
    int first;
    int second;
    int count;
} Bigram;

int replace_and_compact(short* arr, int size, Bigram bigram, int new_token){
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

	char dir[] = "enwik9";
	char out[] = "tokenized_enwik9";
	data = fopen(dir, "r");
    FILE* first_file = fopen("first", "r");
    FILE* second_file = fopen("second", "r");

    int* first = (int*)malloc(2048 * sizeof(int));
    int* second = (int*)malloc(2048 * sizeof(int));

    fread(first, sizeof(int), 2048, first_file);
    fread(second, sizeof(int), 2048, second_file);

    fclose(first_file);
    fclose(second_file);

	__uint8_t* arr = (__uint8_t*)malloc(arr_size * sizeof(__uint8_t));
	fread(arr, sizeof(__uint8_t), arr_size, data);
	fclose(data);

	short* int16_arr = (short*)malloc(arr_size * sizeof(short));
	for (int i = 0; i < arr_size; i++){
		int16_arr[i] = (short)arr[i];
	}
	free(arr);

    for (int i = 0; i < 2048; i++){
        Bigram bigram = {first[i], second[i]};
        int removed = replace_and_compact(int16_arr, arr_size, bigram, 256 + i);
        arr_size -= removed;
        printf("%f%% | %d | %d\n", (float)i/2048 * 100, i, arr_size);
    }
    
	file = fopen(out, "wb");
	fwrite(int16_arr, sizeof(short), arr_size, file);
	fclose(file);

	return 0;
}
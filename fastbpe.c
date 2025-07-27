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
    int used;
} BigramEntry;

BigramEntry table[TABLE_SIZE];

unsigned int hash(int a, int b) {
    return ((a * 2654435761u) ^ (b * 40503u)) % TABLE_SIZE;
}

BigramEntry get_bigram(short* arr, int size){
	for (int i = 0; i < TABLE_SIZE; ++i) {
        table[i].count = 0;
        table[i].used = 0;
    }

    for (int i = 0; i < size - 1; ++i) {
        int a = arr[i];
        int b = arr[i + 1];
        unsigned int h = hash(a, b);

        while (1) {
            if (!table[h].used) {
                table[h].first = a;
                table[h].second = b;
                table[h].count = 1;
                table[h].used = 1;
                break;
            } else if (table[h].first == a && table[h].second == b) {
                table[h].count++;
                break;
            } else {
                h = (h + 1) % TABLE_SIZE;
            }
        }
    }

    int max_count = 0;
    int max_a = 0, max_b = 0;
    for (int i = 0; i < TABLE_SIZE; ++i) {
        if (table[i].used && table[i].count > max_count) {
            max_count = table[i].count;
            max_a = table[i].first;
            max_b = table[i].second;
        }
    }
	return table[hash(max_a, max_b)];
}

int replace_bigram(short* arr, BigramEntry bigram, int size, int new_token){
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

	int first[2048] = {0};
	int second[2048] = {0};

	for (int i = 0; i < 2048; i++){
		BigramEntry bigram = get_bigram(int16_arr, arr_size);
		first[i] = bigram.first;
		second[i] = bigram.second;
		int count = replace_bigram(int16_arr, bigram, arr_size, 256+i);
		remove_empty(int16_arr, arr_size, (arr_size - count));
		arr_size = (arr_size - count);
		int used_count = 0;
		for (int i = 0; i < TABLE_SIZE; i++){
			if (table[i].used == 1){
				used_count += 1;
			}
		}
		printf("%f%% | %d | %d\n", ((float)used_count / TABLE_SIZE) * 100, i, arr_size);
	}
	
	file = fopen(out, "wb");
	fwrite(int16_arr, sizeof(short), arr_size, file);
	fclose(file);

	FILE* first_file = fopen("first", "wb");
	FILE* second_file = fopen("second", "wb");
	fwrite(first, sizeof(int), 2048, first_file);
	fwrite(second, sizeof(int), 2048, second_file);
	fclose(first_file);
	fclose(second_file);

	return 0;
}
// bpe_decode.c
// Simple decoder for the encoder you provided.
// Reads: "tokenized_enwik9" (uint16_ts), "first" (ints), "second" (ints).
// Writes: "decoded_enwik9" (bytes)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int64_t file_size_bytes(FILE *f) {
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    int64_t s = ftell(f);
    rewind(f);
    return s;
}

int main(void) {
    const char encoded_name[] = "60k_enwik9";
    const char first_name[] = "first";
    const char second_name[] = "second";
    const char out_name[] = "decompressed_enwik9";

    // Open first/second and determine number of merges (num_tokens)
    FILE *f_first = fopen(first_name, "rb");
    FILE *f_second = fopen(second_name, "rb");
    if (!f_first || !f_second) {
        fprintf(stderr, "Error: could not open '%s' or '%s'\n", first_name, second_name);
        return 1;
    }
    int64_t first_bytes = file_size_bytes(f_first);
    int64_t second_bytes = file_size_bytes(f_second);
    if (first_bytes < 0 || second_bytes < 0 || first_bytes != second_bytes) {
        fprintf(stderr, "Error: bad size for first/second files\n");
        return 1;
    }
    int num_tokens = (int)(first_bytes / (int)sizeof(int));
    if (num_tokens <= 0) {
        fprintf(stderr, "Error: no tokens found in first/second\n");
        return 1;
    }

    // Read first/second arrays
    int *first = (int*)malloc((size_t)num_tokens * sizeof(int));
    int *second = (int*)malloc((size_t)num_tokens * sizeof(int));
    if (!first || !second) {
        fprintf(stderr, "malloc failed for first/second\n");
        return 1;
    }
    size_t r1 = fread(first, sizeof(int), (size_t)num_tokens, f_first);
    size_t r2 = fread(second, sizeof(int), (size_t)num_tokens, f_second);
    fclose(f_first);
    fclose(f_second);
    if (r1 != (size_t)num_tokens || r2 != (size_t)num_tokens) {
        fprintf(stderr, "Error: could not read first/second arrays fully\n");
        return 1;
    }

    // Read encoded token stream (uint16_ts)
    FILE *f_enc = fopen(encoded_name, "rb");
    if (!f_enc) {
        fprintf(stderr, "Error: could not open encoded file '%s'\n", encoded_name);
        return 1;
    }
    int64_t enc_bytes = file_size_bytes(f_enc);
    if (enc_bytes < 0 || (enc_bytes % sizeof(uint16_t) != 0)) {
        fprintf(stderr, "Error: encoded file has unexpected size\n");
        return 1;
    }
    int enc_size = (int)(enc_bytes / sizeof(uint16_t));
    uint16_t *arr = (uint16_t*)malloc((size_t)enc_size * sizeof(uint16_t));
    if (!arr) {
        fprintf(stderr, "malloc failed for encoded array\n");
        return 1;
    }
    size_t got = fread(arr, sizeof(uint16_t), (size_t)enc_size, f_enc);
    fclose(f_enc);
    if (got != (size_t)enc_size) {
        fprintf(stderr, "Error: could not read encoded array fully\n");
        return 1;
    }

    // Decode: expand merges in reverse order
    for (int i = num_tokens - 1; i >= 0; --i) {
        int merged_token = 256 + i;
        int a = first[i];
        int b = second[i];

        // Count occurrences of merged_token
        int count = 0;
        for (int k = 0; k < enc_size; ++k) {
            if (arr[k] == (uint16_t)merged_token) count++;
        }
        if (count == 0) {
            // nothing to expand for this merge
            printf("iter %d/%d: merged token %d occurrences=0 size=%d\n",
                   i+1, num_tokens, merged_token, enc_size);
            continue;
        }

        // New size after expansion: each occurrence of merged_token becomes two tokens -> +count
        int new_size = enc_size + count;
        uint16_t *new_arr = (uint16_t*)malloc((size_t)new_size * sizeof(uint16_t));
        if (!new_arr) {
            fprintf(stderr, "malloc failed during expansion at iter %d\n", i);
            free(arr);
            free(first);
            free(second);
            return 1;
        }

        // Fill new array by replacing merged_token -> (a,b)
        int write = 0;
        for (int k = 0; k < enc_size; ++k) {
            if (arr[k] == (uint16_t)merged_token) {
                new_arr[write++] = (uint16_t)a;
                new_arr[write++] = (uint16_t)b;
            } else {
                new_arr[write++] = arr[k];
            }
        }
        // Sanity check
        if (write != new_size) {
            fprintf(stderr, "internal error: write (%d) != new_size (%d)\n", write, new_size);
            free(new_arr);
            free(arr);
            free(first);
            free(second);
            return 1;
        }

        // replace arrays
        free(arr);
        arr = new_arr;
        enc_size = new_size;

        printf("iter %d/%d: expanded token %d occurrences=%d new_size=%d\n",
               i+1, num_tokens, merged_token, count, enc_size);
    }

    // Write decoded bytes to file (assume final tokens are 0..255)
    FILE *f_out = fopen(out_name, "wb");
    if (!f_out) {
        fprintf(stderr, "Error: could not open output file '%s'\n", out_name);
        free(arr);
        free(first);
        free(second);
        return 1;
    }
    // convert uint16_ts to bytes and write
    for (int i = 0; i < enc_size; ++i) {
        unsigned char byte = (unsigned char)(arr[i] & 0xFF);
        fwrite(&byte, sizeof(unsigned char), 1, f_out);
    }
    fclose(f_out);

    printf("Decoding complete. Wrote %d bytes to '%s'\n", enc_size, out_name);

    free(arr);
    free(first);
    free(second);
    return 0;
}

// bpe_corrected.c
// Correct BPE with logical adjacency (next non-zero) and incremental bigram updates.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define TABLE_SIZE 90000000

typedef struct {
    int first;
    int second;
    int count;
    int used;
} BigramEntry;

static BigramEntry table[TABLE_SIZE];
static unsigned int hash_list[TABLE_SIZE];
static int hash_list_size = 0;

static inline unsigned int mix32(unsigned int x) {
    // mixing to spread bits (Knuth multiplicative)
    return x * 2654435761u;
}

static inline unsigned int bigram_hash(int a, int b) {
    unsigned int h = mix32((unsigned int)a) ^ (mix32((unsigned int)b) >> 1);
    return h % TABLE_SIZE;
}

// Find existing index, or -1 if not present
static int find_index(int a, int b) {
    unsigned int h = bigram_hash(a, b);
    unsigned int start = h;
    while (1) {
        if (!table[h].used) return -1;
        if (table[h].first == a && table[h].second == b) return (int)h;
        h = h + 1;
        if (h == TABLE_SIZE) h = 0;
        if (h == start) return -1; // table full or not found
    }
}

// Insert or return index for (a,b) — creates entry with count 0 if new
static int insert_or_get(int a, int b) {
    unsigned int h = bigram_hash(a, b);
    while (1) {
        if (!table[h].used) {
            table[h].used = 1;
            table[h].first = a;
            table[h].second = b;
            table[h].count = 0;
            // record used index for scanning
            if (hash_list_size < TABLE_SIZE) hash_list[hash_list_size++] = h;
            return (int)h;
        } else if (table[h].first == a && table[h].second == b) {
            return (int)h;
        } else {
            h = h + 1;
            if (h == TABLE_SIZE) h = 0;
        }
    }
}

static inline void inc_count(int a, int b, int delta) {
    int idx = insert_or_get(a, b);
    table[idx].count += delta;
    if (table[idx].count < 0) table[idx].count = 0;
}

static inline void dec_count_if_exists(int a, int b, int delta) {
    int idx = find_index(a, b);
    if (idx >= 0) {
        table[idx].count -= delta;
        if (table[idx].count < 0) table[idx].count = 0;
    }
}

// Build bigram table from scratch using the "skip zeros" logical adjacency
static void build_table(uint16_t *arr, int size) {
    // clear only used entries
    for (int i = 0; i < hash_list_size; ++i) {
        unsigned int idx = hash_list[i];
        table[idx].used = 0;
        table[idx].count = 0;
    }
    hash_list_size = 0;

    int prev = -1; // holds previous non-zero token value
    for (int i = 0; i < size; ++i) {
        if (arr[i] == 0) continue;
        if (prev != -1) {
            inc_count(prev, arr[i], 1);
        }
        prev = arr[i];
    }
}

// Pick index of highest-count bigram (scan only entries that were created)
static int pick_max_bigram_index(void) {
    int best = -1;
    int best_count = 0;
    for (int i = 0; i < hash_list_size; ++i) {
        unsigned int idx = hash_list[i];
        if (table[idx].used && table[idx].count > best_count) {
            best_count = table[idx].count;
            best = (int)idx;
        }
    }
    return best;
}

// Replace occurrences of bigram (first,second) with new_token.
// This uses logical adjacency (next non-zero) and updates counts incrementally.
// Returns number of replacements performed.
static int replace_and_update(uint16_t *arr, int size, BigramEntry bigram, int new_token) {
    int replaced = 0;

    int prev_idx = -1;       // index of previous non-zero token in array
    int prev_prev_idx = -1;  // index of token before prev_idx (for left bigram updates)

    int i = 0;
    while (i < size) {
        if (arr[i] == 0) { ++i; continue; }
        int curr_idx = i;          // current non-zero position
        if (prev_idx != -1) {
            int a_val = arr[prev_idx];
            int b_val = arr[curr_idx];

            if (a_val == bigram.first && b_val == bigram.second) {
                // find next non-zero after curr_idx
                int j = curr_idx + 1;
                while (j < size && arr[j] == 0) ++j;

                // update counts for left neighbor: (prev_prev, a) -> (prev_prev, new_token)
                if (prev_prev_idx != -1) {
                    dec_count_if_exists(arr[prev_prev_idx], a_val, 1);
                    inc_count(arr[prev_prev_idx], new_token, 1);
                }

                // remove the matched (a,b)
                dec_count_if_exists(a_val, b_val, 1);

                // update counts for right neighbor: (b, next) -> (new_token, next)
                if (j < size) {
                    dec_count_if_exists(b_val, arr[j], 1);
                    inc_count(new_token, arr[j], 1);
                }

                // perform replacement in array: place new_token at prev_idx, zero-out curr_idx
                arr[prev_idx] = (uint16_t)new_token;
                arr[curr_idx] = 0;

                replaced += 1;

                // After replacing, the last non-zero token seen is at prev_idx (new_token).
                // prev_prev_idx remains the same (the token before prev_idx).
                // Move i to j to continue from next non-zero after the removed token.
                i = j;
                // prev_idx stays as prev_idx (index of new_token)
                // prev_prev_idx remains unchanged
                continue;
            } else {
                // no match: shift prev pointers forward
                prev_prev_idx = prev_idx;
                prev_idx = curr_idx;
            }
        } else {
            // first non-zero we've seen
            prev_prev_idx = -1;
            prev_idx = curr_idx;
        }
        ++i;
    }

    return replaced;
}

// Remove zeros from array: shift non-zero tokens left into a new compact array
static void remove_empty(uint16_t *arr, int size, int new_size) {
    uint16_t *tmp = (uint16_t*)malloc(new_size * sizeof(uint16_t));
    if (!tmp) { fprintf(stderr, "malloc failed in remove_empty\n"); exit(1); }
    int w = 0;
    for (int i = 0; i < size && w < new_size; ++i) {
        if (arr[i] != 0) tmp[w++] = arr[i];
    }
    // fill back
    for (int i = 0; i < new_size; ++i) arr[i] = tmp[i];
    free(tmp);
}

int main(void) {
    // initialize table
    for (int i = 0; i < TABLE_SIZE; ++i) { table[i].count = 0; table[i].used = 0; }
    hash_list_size = 0;

    // adjust these values to match your environment (1e9 is huge)
    int arr_size = (int)pow(10, 9);      // input array length (bytes/tokens)
    int cosmetic_arr_size = arr_size;    // logical length excluding removed entries

    const char in_name[] = "enwik9";
    const char out_name[] = "tokenized_enwik9";

    // read file as bytes (0..255), store into int16 array
    FILE *data = fopen(in_name, "rb");
    uint16_t *int16_arr = (uint16_t*)malloc(arr_size * sizeof(uint16_t));
    if (!int16_arr) { fprintf(stderr, "malloc failed\n"); return 1; }

    if (data) {
        uint8_t *tmp = (uint8_t*)malloc(arr_size * sizeof(uint8_t));
        if (!tmp) { fprintf(stderr, "malloc failed\n"); return 1; }
        size_t got = fread(tmp, sizeof(uint8_t), (size_t)arr_size, data);
        fclose(data);
        for (int i = 0; i < arr_size; ++i) {
            int16_arr[i] = (i < (int)got) ? (uint16_t)tmp[i] : 0;
        }
        free(tmp);
    } else {
        // fallback synthetic sample if file not found
        fprintf(stderr, "warning: '%s' not opened. Using synthetic data.\n", in_name);
        for (int i = 0; i < arr_size; ++i) int16_arr[i] = (uint16_t)((i % 255) + 1);
    }

    // number of merges to perform
    int num_tokens = 60000;
    int *first = (int*)malloc(num_tokens * sizeof(int));
    int *second = (int*)malloc(num_tokens * sizeof(int));
    if (!first || !second) { fprintf(stderr, "malloc failed\n"); return 1; }

    // build initial bigram counts (logical adjacency)
    build_table(int16_arr, arr_size);

    for (int iter = 0; iter < num_tokens; ++iter) {
        int best_idx = pick_max_bigram_index();
        if (best_idx < 0) {
            fprintf(stderr, "no bigram available at iter %d, stopping\n", iter);
            num_tokens = iter;
            break;
        }
        BigramEntry chosen = table[best_idx];
        if (chosen.count == 0) {
            fprintf(stderr, "all remaining bigrams have count 0 at iter %d, stopping\n", iter);
            num_tokens = iter;
            break;
        }

        first[iter] = chosen.first;
        second[iter] = chosen.second;
        int new_token = 256 + iter;

        int replaced = replace_and_update(int16_arr, arr_size, chosen, new_token);

        // optional compaction: every N iterations (keeps array smaller)
        // compaction does NOT change logical adjacency because our adjacency is "skip zeros"
        if (iter % 100 == 0) {
            cosmetic_arr_size -= replaced;
            if (cosmetic_arr_size < 0) cosmetic_arr_size = 0;
            remove_empty(int16_arr, arr_size, cosmetic_arr_size);
            arr_size = cosmetic_arr_size;
            // NO rebuild needed — counts remain correct because adjacency is logical.
        } else {
            cosmetic_arr_size -= replaced;
            if (cosmetic_arr_size < 0) cosmetic_arr_size = 0;
        }

        // diagnostic: how many used table slots
        int used_count = 0;
        for (int i = 0; i < hash_list_size; ++i) {
            if (table[hash_list[i]].used) used_count++;
        }
        printf("iter %d: chosen=(%d,%d) count=%d replaced=%d tokens=%d used_table=%.2f%%\n",
               iter+1, chosen.first, chosen.second, chosen.count, replaced,
               cosmetic_arr_size, ((double)used_count / TABLE_SIZE) * 100.0);
    }

    // write outputs (like original)
    FILE *out = fopen(out_name, "wb");
    if (out) {
        fwrite(int16_arr, sizeof(uint16_t), arr_size, out);
        fclose(out);
    } else {
        fprintf(stderr, "warning: could not open output '%s'\n", out_name);
    }

    FILE *f1 = fopen("first", "wb");
    FILE *f2 = fopen("second", "wb");
    if (f1 && f2) {
        fwrite(first, sizeof(int), num_tokens, f1);
        fwrite(second, sizeof(int), num_tokens, f2);
        fclose(f1);
        fclose(f2);
    } else {
        fprintf(stderr, "warning: could not write first/second\n");
    }

    free(int16_arr);
    free(first);
    free(second);
    return 0;
}

#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <stddef.h>
#include <pthread.h>
#include <stdbool.h>


// Data generator structure definition
typedef struct DataGenerator {
    char *url;
    char *output_file;
    size_t dataSize;
    size_t chunk_size;
    size_t currentIndex;
    size_t current_call_number;
    int finished;
    pthread_mutex_t mutex_generator;
} DataGenerator;

// Structure that represents a chunk of work for parallel processing
typedef struct parallel_work_data {
    char *url;
    char *output_file;
    size_t start_byte;
    size_t end_byte;
} parallel_work_data;

// Function prototypes for the data generator operations

/**
 * Initialize the data generator.
 * Returns a pointer to a new DataGenerator or NULL if initialization fails.
 */
DataGenerator* data_generator_init(char *url, char *o_file, size_t dataSize, size_t chunkSize);

/**
 * Fetch the next chunk of work for parallel processing.
 * Returns a pointer to a parallel_work_data or NULL if there's no more work or an error occurs.
 */
parallel_work_data* data_generator_next(DataGenerator *gen);

/**
 * Free the resources associated with the given data generator.
 */
void data_generator_free(DataGenerator *gen);

/**
 * Check if the data generator has finished generating work.
 * Returns true if finished, false otherwise.
 */
bool is_finished(DataGenerator *gen);

#endif // DATA_GENERATOR_H

#include "data_generator.h"
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

// Initialize the data generator
DataGenerator* data_generator_init(char *url, char *o_file, size_t dataSize, size_t chunkSize) {
    DataGenerator *gen = malloc(sizeof(DataGenerator));
    if (!gen) {
        return NULL; // Allocation failed
    }
    gen->url = strdup(url);
    if (!gen->url) {
        free(gen);
        return NULL; // strdup failed
    }
    gen->output_file = strdup(o_file);
    if (!gen->output_file) {
        free(gen->url);
        free(gen);
        return NULL; // strdup failed
    }
    gen->dataSize = dataSize;
    gen->chunk_size = chunkSize;
    gen->currentIndex = 0;
    gen->current_call_number = 0;
    gen->finished = 0;

    // Initialize mutex
    if (pthread_mutex_init(&gen->mutex_generator, NULL) != 0) {
        free(gen->url);
        free(gen->output_file);
        free(gen);
        return NULL; // Mutex initialization failed
    }
    return gen;
}

// Fetch next chunk of work
parallel_work_data* data_generator_next(DataGenerator *gen) {
    if (gen == NULL) {
        return NULL;
    }
    pthread_mutex_lock(&gen->mutex_generator);
    if (gen->currentIndex >= gen->dataSize) {
        gen->finished = 1;
        pthread_mutex_unlock(&gen->mutex_generator);
        return NULL; // No more chunks to process
    }
    parallel_work_data *work_data = malloc(sizeof(parallel_work_data));
    if (!work_data) {
        pthread_mutex_unlock(&gen->mutex_generator);
        return NULL; // Allocation failed
    }
    work_data->url = strdup(gen->url);
    if (!work_data->url) {
        free(work_data);
        pthread_mutex_unlock(&gen->mutex_generator);
        return NULL; // strdup failed
    }
    work_data->output_file = malloc(strlen(gen->output_file) + 1);
    if (!work_data->output_file) {
        free(work_data->url);
        free(work_data);
        pthread_mutex_unlock(&gen->mutex_generator);
        return NULL; // malloc failed
    }
    strcpy(work_data->output_file, gen->output_file);

    work_data->start_byte = gen->currentIndex;
    work_data->end_byte = gen->currentIndex + gen->chunk_size - 1;
    if (work_data->end_byte >= gen->dataSize) {
        work_data->end_byte = gen->dataSize - 1;
    }

    gen->currentIndex += gen->chunk_size;
    gen->current_call_number++;
    pthread_mutex_unlock(&gen->mutex_generator);
    return work_data;
}

// Free the data generator
void data_generator_free(DataGenerator *gen) {
    if (gen) {
        free(gen->url);
        free(gen->output_file);
        pthread_mutex_destroy(&gen->mutex_generator);
        free(gen);
    }
}

bool is_finished(DataGenerator *gen) {
    if (gen == NULL) {
        return false;
    }
    pthread_mutex_lock(&gen->mutex_generator);
    bool finished = (gen->finished == 1);
    pthread_mutex_unlock(&gen->mutex_generator);
    return finished;
}

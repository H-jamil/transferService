#include "data_generator.h"
#include <stdbool.h>
#include <string.h>


DataGenerator* data_generator_init(char *url, char *o_file, size_t dataSize, size_t chunkSize) {
    DataGenerator *gen = malloc(sizeof(DataGenerator));
    if (!gen) {
        return NULL; // Allocation failed
    }
    gen->url = strdup(url); // Make a copy of the url string
    gen->output_file = strdup(o_file); // Make a copy of the output file string
    gen->dataSize = dataSize;
    gen->chunk_size = chunkSize;
    gen->currentIndex = 0;
    gen->current_call_number = 0;
    gen->finished = 0;
    // Initialize mutexes
    pthread_mutex_init(&gen->mutex_generator, NULL);
    pthread_mutex_init(&gen->mutex_outfile_file, NULL);
    pthread_mutex_init(&gen->mutex_generator_finished, NULL);
    return gen;
}

// Fetch next chunk of work
parallel_work_data* data_generator_next(DataGenerator *gen) {
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

    work_data->url = strdup(gen->url); // Copy the URL
    work_data->output_file = malloc(strlen(gen->output_file)); // Add space for potential multiplier
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
    free(gen->url);
    free(gen->output_file);
    pthread_mutex_destroy(&gen->mutex_generator);
    pthread_mutex_destroy(&gen->mutex_outfile_file);
    free(gen);
}

bool is_finished(DataGenerator *gen){
    if (gen == NULL) {
        return true;
    }
    pthread_mutex_lock(&gen->mutex_generator_finished);
    bool finished = (gen->finished == 1);
    pthread_mutex_unlock(&gen->mutex_generator_finished);
    return finished;
}

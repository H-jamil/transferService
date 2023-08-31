#include "thread.h"
#include "data_generator.h"
extern FILE* logFile;
extern pthread_mutex_t logMutex;


void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value) {
    for(int i = 0; i < active_parallel_value; i++) {
        resume_parallel_worker(&thread_data[i]);
    }
    for(int i = active_parallel_value; i < MAX_PARALLELISM; i++) {
        pause_parallel_worker(&thread_data[i]);
    }
}

void set_parallel_value(ConcurrencyWorkerData* data, int value) {
    pthread_mutex_lock(&data->parallel_value_mutex);
    data->parallel_value = value;
    pthread_mutex_unlock(&data->parallel_value_mutex);
}

int get_parallel_value(ConcurrencyWorkerData* data) {
    pthread_mutex_lock(&data->parallel_value_mutex);
    int value = data->parallel_value;
    pthread_mutex_unlock(&data->parallel_value_mutex);
    return value;
}

void pause_parallel_worker(ParallelWorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 1;
    pthread_mutex_unlock(&data->pause_mutex);
}

void resume_parallel_worker(ParallelWorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 0;
    pthread_cond_signal(&data->pause_cond);
    pthread_mutex_unlock(&data->pause_mutex);
}


void* ParallelThreadFunc(void* arg) {
    ParallelWorkerData* data = (ParallelWorkerData*) arg;
    parallel_work_data *chunk;
    // Following condition is required for thread to be active. If data->active = 0
    // the thread will shut itself down
    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        // Parallel Thread is Paused below if data->paused is true
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);
        // Parallel Thread runs below if data->paused is false
        time_t now;
        // time(&now);
        // printf("Parent ID : %d thread ID: %d, Time: %s", data->parent_id, data->id, ctime(&now));
        // Busy waiting loop for approximately 1 second
        struct timespec start, current;
        clock_gettime(CLOCK_MONOTONIC, &start);
        do {
            // the download part of the code from a generator
            chunk = data_generator_next(data->data_generator);
            time(&now);
            // printf("Parent ID : %d thread ID: %d, work url : %s , start_byte :%ld \n", data->parent_id, data->id, chunk->url, chunk->start_byte);
            pthread_mutex_lock(&logMutex);
            fprintf(logFile,"Parent ID : %d thread ID: %d, work url : %s , start_byte :%ld \n", data->parent_id, data->id, chunk->url, chunk->start_byte);
            pthread_mutex_unlock(&logMutex);

            sleep(1);
            clock_gettime(CLOCK_MONOTONIC, &current);
        } while ((current.tv_sec - start.tv_sec) + (current.tv_nsec - start.tv_nsec) / 1e9 < 1.0);
    }

    // printf("Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id, data->id);
    pthread_mutex_unlock(&logMutex);
    fprintf(logFile,"Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id, data->id);
    pthread_mutex_unlock(&logMutex);

    return NULL;
}

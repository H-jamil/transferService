#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include "queue.h"
#include "data_generator.h"

#define MAX_CONCURRENCY 16
#define MAX_PARALLELISM 16
#define UPDATE_TIME 3
extern FILE* logFile;
extern pthread_mutex_t logMutex;

typedef struct ParallelWorkerData{
    int id;
    int active;
    DataGenerator* d_generator;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parent_id;
} ParallelWorkerData;

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

void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value) {
    for(int i = 0; i < active_parallel_value; i++) {
        resume_parallel_worker(&thread_data[i]);
    }
    for(int i = active_parallel_value; i < MAX_PARALLELISM; i++) {
        pause_parallel_worker(&thread_data[i]);
    }
}


void terminate_parallel_worker(pthread_t thread) {
    pthread_cancel(thread);
}


void* ParallelThreadFunc(void* arg) {
    ParallelWorkerData* data = (ParallelWorkerData*) arg;
    parallel_work_data *chunk;
    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);
        struct timespec start, current;
        clock_gettime(CLOCK_MONOTONIC, &start);
        do {
            if((chunk = data_generator_next(data->d_generator))!=NULL){
                    pthread_mutex_lock(&logMutex);
                    fprintf(logFile,"Parent ID : %d thread ID: %d, file name: %s , start_byte :%ld \n", data->parent_id, data->id, chunk->output_file, chunk->start_byte);                    pthread_mutex_unlock(&logMutex);
                    free(chunk->url);
                    free(chunk->output_file);
                    free(chunk);
              }
            else{
                break;
            }
            // printf("Parent ID : %d thread ID: %d working ..\n", data->parent_id, data->id);
            sleep(1);
            clock_gettime(CLOCK_MONOTONIC, &current);
        } while ((current.tv_sec - start.tv_sec) + (current.tv_nsec - start.tv_nsec) / 1e9 < 1.0);
    }
    pthread_mutex_lock(&logMutex);
    fprintf(logFile,"Parent ID : %d thread ID: %d, exiting\n", data->parent_id, data->id);
    pthread_mutex_unlock(&logMutex);
    return NULL;
}

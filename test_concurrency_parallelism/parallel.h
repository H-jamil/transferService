#ifndef PARALLEL_WORKER_H
#define PARALLEL_WORKER_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>


#define MAX_CONCURRENCY 4
#define MAX_PARALLELISM 4
#define UPDATE_TIME 3

typedef struct ParallelWorkerData {
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parent_id;
} ParallelWorkerData;

void pause_parallel_worker(ParallelWorkerData* data);
void resume_parallel_worker(ParallelWorkerData* data);
void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value);
void terminate_parallel_worker(pthread_t thread);
void* ParallelThreadFunc(void* arg);

#endif /* PARALLEL_WORKER_H */

#ifndef THREAD_H
#define THREAD_H
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>


#define MAX_CONCURRENCY 4
#define MAX_PARALLELISM 4
#define UPDATE_TIME 3

typedef struct ParallelWorkerData{
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parent_id;
} ParallelWorkerData;

typedef struct ConcurrencyWorkerData{
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parallel_value;
    pthread_mutex_t parallel_value_mutex;
    ParallelWorkerData* thread_data;
    int global_concurrency_value;
    pthread_mutex_t global_concurrency_value_mutex;
} ConcurrencyWorkerData;

void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value);
void set_parallel_value(ConcurrencyWorkerData* data, int value);
int get_parallel_value(ConcurrencyWorkerData* data);
void pause_parallel_worker(ParallelWorkerData* data);
void resume_parallel_worker(ParallelWorkerData* data);
void* ParallelThreadFunc(void* arg);

void set_concurrent_value(ConcurrencyWorkerData* data, int value);
int get_concurrent_value(ConcurrencyWorkerData* data);
void adjust_concurrency_workers(ConcurrencyWorkerData* data);
void pause_concurrency_worker(ConcurrencyWorkerData* data);
void resume_concurrency_worker(ConcurrencyWorkerData* data);
void* ConcurrencyThreadFunc(void* arg);
#endif

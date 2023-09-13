#ifndef CONCURRENCY_WORKER_H
#define CONCURRENCY_WORKER_H

#include <string.h>
#include "parallel.h"
#include "data_generator.h"
#include "queue.h"
#include <signal.h>

extern int current_concurrency;
extern pthread_mutex_t concurrency_mutex;
extern int current_parallelism;
extern pthread_mutex_t parallelism_mutex;
extern FILE* logFile;
extern pthread_mutex_t logMutex;

typedef struct ConcurrencyWorkerData{
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    Queue* files_need_to_be_downloaded;
    ParallelWorkerData* thread_data;
} ConcurrencyWorkerData;

void set_concurrent_value(int value);
int get_concurrent_value();

void set_parallel_value(int value);

int get_parallel_value();

void pause_concurrency_worker(ConcurrencyWorkerData* data);
void resume_concurrency_worker(ConcurrencyWorkerData* data);

void* ConcurrencyThreadFunc(void* arg);

#endif /* CONCURRENCY_WORKER_H */

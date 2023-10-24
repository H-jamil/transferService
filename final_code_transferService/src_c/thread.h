#ifndef THREAD_H
#define THREAD_H
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <curl/curl.h>
#include "/home/jamilm/libzmq/include/zmq.h"
// #include <zmq.h>
#include "data_generator.h"
#include "queue.h"


#define MAX_CONCURRENCY 17
#define MAX_PARALLELISM 17
#define UPDATE_TIME 3

typedef struct ParallelWorkerData{
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parent_id;
    DataGenerator* data_generator;
} ParallelWorkerData;

typedef struct ConcurrencyWorkerData{
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    Queue *files_need_to_be_downloaded;
    Queue *files_downloaded;
    int chunk_size;
    pthread_mutex_t parallel_value_mutex;
    ParallelWorkerData* thread_data;

} ConcurrencyWorkerData;

void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value);
void set_parallel_value(int value);
int get_parallel_value();
void pause_parallel_worker(ParallelWorkerData* data);
void resume_parallel_worker(ParallelWorkerData* data);
size_t write_callback_parallel(char *ptr, size_t size, size_t nmemb, void *userdata);
void download_part(parallel_work_data *data);
void* ParallelThreadFunc(void* arg);

char* extract_filename(const char* url);
double get_file_size_from_url(const char *url);
Queue* get_generator_queue(Queue *files_need_to_be_downloaded,int CHUNK_SIZE);
void set_concurrent_value(int value);
int get_concurrent_value();
void pause_concurrency_worker(ConcurrencyWorkerData* data);
void resume_concurrency_worker(ConcurrencyWorkerData* data);
void* ConcurrencyThreadFunc(void* arg);
#endif

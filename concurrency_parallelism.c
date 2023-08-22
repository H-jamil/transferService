#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define MAX_PARALLELISM 4
#define MAX_CONCURRENCY 4

typedef struct {
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parallel_value;
} ConcurrencyWorkerData;

typedef struct {
    int id;
    int active;
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

void pause_concurrent_worker(ConcurrencyWorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 1;
    pthread_mutex_unlock(&data->pause_mutex);
}

void resume_concurrent_worker(ConcurrencyWorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 0;
    pthread_cond_signal(&data->pause_cond);
    pthread_mutex_unlock(&data->pause_mutex);
}


void* Parallel_thread(void* arg) {
    ParallelWorkerData* data = (ParallelWorkerData*) arg;
    clock_t start_time, end_time;

    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);

        time_t now;
        time(&now);
        printf("Parent ID : %d thread ID: %d, Time: %s", data->parent_id,data->id, ctime(&now));

        // Busy-wait loop for about 1 second
        start_time = clock();
        do {
            end_time = clock();
        } while((double)(end_time - start_time) / CLOCKS_PER_SEC < 1.0);
    }

    return NULL;
}


void* Concurrency_thread(void* arg){
  ConcurrencyWorkerData* data=(ConcurrencyWorkerData*) arg;
  pthread_t threads[MAX_PARALLELISM];
  ParallelWorkerData thread_data[MAX_PARALLELISM];
  for(int i = 0; i < MAX_PARALLELISM; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;
        thread_data[i].parent_id=data->id;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, Parallel_thread, &thread_data[i]);
    }
  printf("Concurrent Thread %d creating parallel threads",data->id);
  while(1){
    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);
      }

  }
  return NULL;
}

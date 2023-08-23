#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define MAX_PARALLELISM 4
#define MAX_CONCURRENCY 4
#define UPDATE_TIME 3

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
    printf("Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id,data->id);
    return NULL;
}


void* Concurrency_thread(void* arg){
  ConcurrencyWorkerData* data=(ConcurrencyWorkerData*) arg;
  int active_parallel_value=0;
  int old_active_parallel_value=-1;
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
  printf("Concurrent Thread %d creating all parallel threads (paused)\n",data->id);
  while(data->active) {
      pthread_mutex_lock(&data->pause_mutex);
      while(data->paused) {
          for(int i = 0; i < MAX_PARALLELISM; i++) {
                pause_parallel_worker(&thread_data[i]);
            }
          pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
      }
      active_parallel_value=data->parallel_value;
      pthread_mutex_unlock(&data->pause_mutex);
      if (active_parallel_value==old_active_parallel_value){
        sleep(UPDATE_TIME);
        old_active_parallel_value=active_parallel_value;
        continue;
      }
      else if (active_parallel_value==MAX_PARALLELISM){
        for(int i = 0; i < MAX_PARALLELISM; i++) {
        resume_parallel_worker(&thread_data[i]);
        }
        sleep(UPDATE_TIME);
        old_active_parallel_value=active_parallel_value;
        continue;
      }
      else {
          for(int i = 0; i < active_parallel_value; i++) {
          resume_parallel_worker(&thread_data[i]);
        }
          for(int i =active_parallel_value ; i < MAX_PARALLELISM; i++) {
            pause_parallel_worker(&thread_data[i]);
          }
        sleep(UPDATE_TIME);
        old_active_parallel_value=active_parallel_value;
        continue;
        }
    }
  printf("Concurrent Thread %d shutting off all parallel threads\n",data->id);
  for(int i = 0; i < MAX_PARALLELISM; i++) {
      thread_data[i].active = 0;
    }
  for(int i = 0; i < MAX_PARALLELISM; i++) {
      pthread_join(threads[i], NULL);
      pthread_mutex_destroy(&thread_data[i].pause_mutex);
      pthread_cond_destroy(&thread_data[i].pause_cond);
    }

  return NULL;
}

int main() {
  pthread_t threads[MAX_CONCURRENCY];
  ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
  // Initialize and start concurrent threads
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        thread_data[i].parallel_value=0;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, Concurrency_thread, &thread_data[i]);
    }
  printf("Main Thread finished creating concurrent threads \n");
  printf("Main Thread change parallelism to Max \n");
  //  change concurrent threads parallel count
  for (int i=0;i<MAX_CONCURRENCY;i++){
    thread_data[i].parallel_value=MAX_PARALLELISM;
  }
  sleep(2*UPDATE_TIME);
  //  change concurrent threads parallel count to 2
  printf("Main Thread change parallelism to 2 \n");
  for (int i=0;i<MAX_CONCURRENCY;i++){
    thread_data[i].parallel_value=2;
  }
  sleep(2*UPDATE_TIME);
  //  change concurrent threads parallel count to 2
  printf("Main Thread change parallelism to 3 \n");
  for (int i=0;i<MAX_CONCURRENCY;i++){
    thread_data[i].parallel_value=3;
  }
  sleep(2*UPDATE_TIME);
  //  change concurrent threads parallel count
  for (int i=0;i<MAX_CONCURRENCY;i++){
    thread_data[i].parallel_value=MAX_PARALLELISM;
  }
  sleep(2*UPDATE_TIME);
// Stop all threads
  printf("Main Thread shutting concurrent threads \n");
  for(int i = 0; i < MAX_CONCURRENCY; i++) {
      thread_data[i].active = 0;
  }

  for(int i = 0; i < MAX_CONCURRENCY; i++) {
      pthread_join(threads[i], NULL);
      pthread_mutex_destroy(&thread_data[i].pause_mutex);
      pthread_cond_destroy(&thread_data[i].pause_cond);
  }

  printf("All threads have been stopped\n");
  return 0;
}

#include "thread.h"
#include <string.h>
extern int current_concurrency;
extern pthread_mutex_t concurrency_mutex;

extern FILE* logFile;
extern pthread_mutex_t logMutex;


char* extract_filename(const char* url) {
    char *last_slash = strrchr(url, '/');
    if (last_slash) {
        return last_slash + 1;  // Move past the '/'
    }
    return url;  // Return NULL if no slash found
}

void set_concurrent_value(int value) {
    pthread_mutex_lock(&concurrency_mutex);
    current_concurrency = value;
    pthread_mutex_unlock(&concurrency_mutex);
}

int get_concurrent_value() {
    pthread_mutex_lock(&concurrency_mutex);
    int value = current_concurrency;
    pthread_mutex_unlock(&concurrency_mutex);
    return value;
}


void pause_concurrency_worker(ConcurrencyWorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 1;
    for (int i = 0; i < MAX_PARALLELISM; i++) {
        pause_parallel_worker(&data->thread_data[i]);
    }
    pthread_mutex_unlock(&data->pause_mutex);
}

void resume_concurrency_worker(ConcurrencyWorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    if(data->paused) {
        data->paused = 0;
        pthread_cond_signal(&data->pause_cond);
    }
    pthread_mutex_unlock(&data->pause_mutex);
    int active_parallel_value = get_parallel_value(data);
    adjust_parallel_workers(data->thread_data, active_parallel_value);
}


void* ConcurrencyThreadFunc(void* arg) {
    ConcurrencyWorkerData* data = (ConcurrencyWorkerData*) arg;
    int active_parallel_value = 0;
    int old_active_parallel_value = -1;
    pthread_t threads[MAX_PARALLELISM];
    ParallelWorkerData thread_data[MAX_PARALLELISM];
    DataGenerator *gen = data_generator_init(data->file_name, extract_filename(data->file_name), 10000000, 10000);
    for(int i = 0; i < MAX_PARALLELISM; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;
        thread_data[i].parent_id = data->id;
        thread_data[i].data_generator=gen;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, ParallelThreadFunc, &thread_data[i]);
    }

    data->thread_data = thread_data;

    // printf("Concurrent Thread %d creating all parallel threads (paused)\n", data->id);
    pthread_mutex_lock(&logMutex);
    fprintf(logFile, "Concurrent Thread %d creating all parallel threads (paused)\n", data->id);
    pthread_mutex_unlock(&logMutex);

    // Following condition is required for thread to be active. If data->active = 0
    // the thread will shut itself down
    while(data->active) {
        pthread_mutex_lock(&concurrency_mutex);
        // Concurrency Thread is Paused below if data->id >= current_concurrency is true
        if (data->id >= current_concurrency) {
            pthread_mutex_unlock(&concurrency_mutex);
            pause_concurrency_worker(data);
            sleep(UPDATE_TIME);
            continue;
            // Concurrent Thread runs below if above condition is false
        } else {
            pthread_mutex_unlock(&concurrency_mutex);
            resume_concurrency_worker(data);
        }
        active_parallel_value = get_parallel_value(data);

        if (active_parallel_value != old_active_parallel_value) {
            adjust_parallel_workers(thread_data, active_parallel_value);
            old_active_parallel_value = active_parallel_value;
        }
        // adjust_concurrency_workers(data);  // Adjust concurrency based on global value
        sleep(UPDATE_TIME);
    }
    pthread_mutex_lock(&logMutex);
    fprintf(logFile,"Concurrent Thread %d shutting off all parallel threads\n", data->id);
    pthread_mutex_unlock(&logMutex);

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

#include "concurrency.h"

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


void set_parallel_value(int value) {
    pthread_mutex_lock(&parallelism_mutex);
    current_parallelism = value;
    pthread_mutex_unlock(&parallelism_mutex);
}


int get_parallel_value() {
    pthread_mutex_lock(&parallelism_mutex);
    int value = current_parallelism;
    pthread_mutex_unlock(&parallelism_mutex);
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
    ParallelWorkerData* thread_data = malloc(MAX_PARALLELISM * sizeof(ParallelWorkerData));
    if(!thread_data) {
        perror("Failed to allocate memory for thread_data");
        return NULL;
    }
    DataGenerator *gen;
    if ((gen= queue_pop(data->files_need_to_be_downloaded)) == NULL) {
        pthread_mutex_lock(&logMutex);
        fprintf(logFile, "Concurrent Thread %d shutting down because there are no files to download\n", data->id);
        pthread_mutex_unlock(&logMutex);
        return NULL;
      }
    for(int i = 0; i < MAX_PARALLELISM; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;
        thread_data[i].parent_id = data->id;
        thread_data[i].generator = gen;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, ParallelThreadFunc, &thread_data[i]);
    }
    data->thread_data = thread_data;
    pthread_mutex_lock(&logMutex);
    fprintf(logFile, "Concurrent Thread %d creating all parallel threads (status: paused)\n", data->id);
    pthread_mutex_unlock(&logMutex);
    while(data->active) {
        pthread_mutex_lock(&concurrency_mutex);
        if (data->id >= current_concurrency) {
            pthread_mutex_unlock(&concurrency_mutex);
            pause_concurrency_worker(data);
            // sleep(UPDATE_TIME);
            sleep(1);
            continue;
        }
        pthread_mutex_unlock(&concurrency_mutex);
        if (is_finished(gen)) {
            if ((gen= queue_pop(data->files_need_to_be_downloaded)) == NULL) {
                pthread_mutex_lock(&logMutex);
                fprintf(logFile, "Concurrent Thread %d shutting down because there are no files to download\n", data->id);
                pthread_mutex_unlock(&logMutex);
                return NULL;
              }
            for(int i = 0; i < MAX_PARALLELISM; i++) {
                thread_data[i].generator = gen;
                }
            data->thread_data = thread_data;
        }
        resume_concurrency_worker(data);
        active_parallel_value = get_parallel_value();
        if (active_parallel_value != old_active_parallel_value) {
            adjust_parallel_workers(thread_data, active_parallel_value);
            old_active_parallel_value = active_parallel_value;
        }
        // sleep(UPDATE_TIME);
        sleep(1);
    }
    for(int i = 0; i < MAX_PARALLELISM; i++) {
        thread_data[i].active = 0;
    }
    for(int i = 0; i < MAX_PARALLELISM; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
    }
    pthread_mutex_lock(&logMutex);
    fprintf(logFile,"Concurrent Thread %d shutdown all of its parallel threads\n", data->id);
    pthread_mutex_unlock(&logMutex);
    free(thread_data);
    return NULL;
}

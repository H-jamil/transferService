#include "thread.h"


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
}


void* ConcurrencyThreadFunc(void* arg) {
    ConcurrencyWorkerData* data = (ConcurrencyWorkerData*) arg;
    int active_parallel_value = 0;
    int old_active_parallel_value = -1;
    pthread_t threads[MAX_PARALLELISM];
    ParallelWorkerData thread_data[MAX_PARALLELISM];

    for(int i = 0; i < MAX_PARALLELISM; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;
        thread_data[i].parent_id = data->id;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, ParallelThreadFunc, &thread_data[i]);
    }

    data->thread_data = thread_data;

    printf("Concurrent Thread %d creating all parallel threads (paused)\n", data->id);

    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);

        active_parallel_value = get_parallel_value(data);

        if (active_parallel_value != old_active_parallel_value) {
            adjust_parallel_workers(thread_data, active_parallel_value);
            old_active_parallel_value = active_parallel_value;
        }

        sleep(UPDATE_TIME);
    }

    printf("Concurrent Thread %d shutting off all parallel threads\n", data->id);
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

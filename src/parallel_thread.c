#include "thread.h"


void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value) {
    for(int i = 0; i < active_parallel_value; i++) {
        resume_parallel_worker(&thread_data[i]);
    }
    for(int i = active_parallel_value; i < MAX_PARALLELISM; i++) {
        pause_parallel_worker(&thread_data[i]);
    }
}

void set_parallel_value(ConcurrencyWorkerData* data, int value) {
    pthread_mutex_lock(&data->parallel_value_mutex);
    data->parallel_value = value;
    pthread_mutex_unlock(&data->parallel_value_mutex);
}

int get_parallel_value(ConcurrencyWorkerData* data) {
    pthread_mutex_lock(&data->parallel_value_mutex);
    int value = data->parallel_value;
    pthread_mutex_unlock(&data->parallel_value_mutex);
    return value;
}

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

void* ParallelThreadFunc(void* arg) {
    ParallelWorkerData* data = (ParallelWorkerData*) arg;
    // Following condition is required for thread to be active. If data->active = 0
    // the thread will shut itself down
    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        // Parallel Thread is Paused below if data->paused is true
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);
        // Parallel Thread runs below if data->paused is false
        time_t now;
        time(&now);
        printf("Parent ID : %d thread ID: %d, Time: %s", data->parent_id, data->id, ctime(&now));
        sleep(1);
    }

    printf("Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id, data->id);
    return NULL;
}

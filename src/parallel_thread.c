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

    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);

        time_t now;
        time(&now);
        printf("Parent ID : %d thread ID: %d, Time: %s", data->parent_id, data->id, ctime(&now));

        sleep(1);
    }

    printf("Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id, data->id);
    return NULL;
}

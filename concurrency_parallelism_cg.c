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
    int parent_id;
} ParallelWorkerData;

typedef struct {
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
    int parallel_value;
    pthread_mutex_t parallel_value_mutex;
    ParallelWorkerData* thread_data;
} ConcurrencyWorkerData;

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

int main() {
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    // Initialize and start concurrent threads
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        thread_data[i].parallel_value = 0;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_mutex_init(&thread_data[i].parallel_value_mutex, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }

    printf("Main Thread change parallelism to Max \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], MAX_PARALLELISM);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 2 \n");
    for (int i = 0; i < 2; i++) {
        pause_concurrency_worker(&thread_data[i]);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change parallelism to 1 \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], 1);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 4 \n");
    for (int i = 0; i < 2; i++) {
        resume_concurrency_worker(&thread_data[i]);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change parallelism to 3 \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], 3);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change parallelism to MAX \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], MAX_PARALLELISM);
    }
    sleep(2*UPDATE_TIME);

    printf("Main thread is shutting off all concurrent threads\n");
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].active = 0;
    }

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
        pthread_mutex_destroy(&thread_data[i].parallel_value_mutex);
    }

    printf("All done\n");
    return 0;
}

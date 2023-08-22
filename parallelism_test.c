#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 5

typedef struct {
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    int paused;
} WorkerData;

void* worker_thread(void* arg) {
    WorkerData* data = (WorkerData*) arg;
    clock_t start_time, end_time;

    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);

        time_t now;
        time(&now);
        printf("ID: %d, Time: %s", data->id, ctime(&now));

        // Busy-wait loop for about 1 second
        start_time = clock();
        do {
            end_time = clock();
        } while((double)(end_time - start_time) / CLOCKS_PER_SEC < 1.0);
    }

    return NULL;
}

void pause_worker(WorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 1;
    pthread_mutex_unlock(&data->pause_mutex);
}

void resume_worker(WorkerData* data) {
    pthread_mutex_lock(&data->pause_mutex);
    data->paused = 0;
    pthread_cond_signal(&data->pause_cond);
    pthread_mutex_unlock(&data->pause_mutex);
}

int main() {
    pthread_t threads[NUM_THREADS];
    WorkerData thread_data[NUM_THREADS];

    // Initialize and start threads
    for(int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }

    sleep(5);

    // Pause the first 2 threads
    printf("\nPausing first 2 threads\n");
    for(int i = 0; i < 2; i++) {
        pause_worker(&thread_data[i]);
    }

    sleep(5);

    // Resume the first 2 threads
    printf("\nResuming paused threads\n");
    for(int i = 0; i < 2; i++) {
        resume_worker(&thread_data[i]);
    }

    sleep(5);

    // Stop all threads
    for(int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].active = 0;
    }

    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
    }

    printf("All threads have been stopped\n");
    return 0;
}

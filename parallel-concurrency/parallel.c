#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <curl/curl.h>
#include "data_generator.h"

#define MAX_CONCURRENCY 4
#define MAX_PARALLELISM 4
#define UPDATE_TIME 3


typedef struct ParallelWorkerData{
    int id;
    int active;
    pthread_mutex_t pause_mutex;
    pthread_cond_t pause_cond;
    DataGenerator* generator;
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

void adjust_parallel_workers(ParallelWorkerData* thread_data, int active_parallel_value) {
    for(int i = 0; i < active_parallel_value; i++) {
        resume_parallel_worker(&thread_data[i]);
    }
    for(int i = active_parallel_value; i < MAX_PARALLELISM; i++) {
        pause_parallel_worker(&thread_data[i]);
    }
}

size_t write_callback_parallel(char *ptr, size_t size, size_t nmemb, void *userdata) {
    parallel_work_data *data = (parallel_work_data *)userdata;
    size_t actual_size = size * nmemb;
    FILE *fp = fopen(data->output_file, "r+b");
    if (!fp) {
        fp = fopen(data->output_file, "wb");
        if (!fp) {
            return -1;  // Return -1 if both attempts to open the file fail
        }
    }
    fseek(fp, data->start_byte, SEEK_SET);
    fwrite(ptr, 1, actual_size, fp);
    fclose(fp);

    data->start_byte += actual_size; // Update the start byte for the next chunk migh be redundant

    return actual_size;
}
size_t write_callback_parallel_no_disk(char *ptr, size_t size, size_t nmemb, void *userdata) {
    parallel_work_data *data = (parallel_work_data *)userdata;
    size_t actual_size = size * nmemb;
    // Update the start_byte value
    data->start_byte += actual_size;
    return actual_size;
}

void download_part(parallel_work_data *data) {
    CURL *curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, data->url);
        char range[50];
        sprintf(range, "%ld-%ld", data->start_byte, data->end_byte);
        curl_easy_setopt(curl, CURLOPT_RANGE, range);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback_parallel);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, data);
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }
        curl_easy_cleanup(curl);
    }
}


void terminate_parallel_worker(pthread_t thread) {
    pthread_cancel(thread);
}

void* ParallelThreadFunc(void* arg) {
    ParallelWorkerData* data = (ParallelWorkerData*) arg;
    parallel_work_data *chunk;
    while(data->active) {
        pthread_mutex_lock(&data->pause_mutex);
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);
        struct timespec start, current;
        clock_gettime(CLOCK_MONOTONIC, &start);
        // why following loop is required? Can you not use it?
        do {
            if((chunk = data_generator_next(data->generator))!=NULL){
                // printf("Parent ID : %d thread ID: %d, work file: %s, start byte: %ld, end byte: %ld\n", data->parent_id, data->id, chunk->output_file, chunk->start_byte, chunk->end_byte);
                download_part(chunk);
                free(chunk);
            }
            else{
                break;
            }
            clock_gettime(CLOCK_MONOTONIC, &current);
        } while ((current.tv_sec - start.tv_sec) + (current.tv_nsec - start.tv_nsec) / 1e9 < 1.0);

    }
    printf("Parent ID : %d thread ID: %d, exiting\n", data->parent_id, data->id);
    return NULL;
}

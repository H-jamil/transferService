#include "thread.h"
#include "data_generator.h"
extern FILE* logFile;
extern pthread_mutex_t logMutex;

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

size_t write_callback_parallel(char *ptr, size_t size, size_t nmemb, void *userdata) {
    parallel_work_data *data = (parallel_work_data *)userdata;
    size_t actual_size = size * nmemb;
    // printf("Writing %ld bytes to file %s\n", actual_size, data->output_file);
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

void* ParallelThreadFunc(void* arg) {
    ParallelWorkerData* data = (ParallelWorkerData*) arg;
    parallel_work_data *chunk;
    // Following condition is required for thread to be active. If data->active = 0
    // the thread will shut itself down
    while(data->active) {
    // while(active_status) {
        pthread_mutex_lock(&data->pause_mutex);
        // Parallel Thread is Paused below if data->paused is true
        while(data->paused) {
            pthread_cond_wait(&data->pause_cond, &data->pause_mutex);
        }
        pthread_mutex_unlock(&data->pause_mutex);
        // Parallel Thread runs below if data->paused is false
        time_t now;
        // time(&now);
        // printf("Parent ID : %d thread ID: %d, Time: %s", data->parent_id, data->id, ctime(&now));
        // Busy waiting loop for approximately 1 second
        struct timespec start, current;
        clock_gettime(CLOCK_MONOTONIC, &start);
        do {
            // the download part of the code from a generator

            if((chunk = data_generator_next(data->data_generator))!=NULL){
                sleep(0.2);
                // download_par1(chunk);
            }
            else{
                // data->active=0;
                break;
            }
            time(&now);
            // printf("Parent ID : %d thread ID: %d, work url : %s , start_byte :%ld \n", data->parent_id, data->id, chunk->url, chunk->start_byte);
            // pthread_mutex_lock(&logMutex);
            // fprintf(logFile,"Parent ID : %d thread ID: %d, file name: %s , start_byte :%ld \n", data->parent_id, data->id, chunk->output_file, chunk->start_byte);
            // pthread_mutex_unlock(&logMutex);
            clock_gettime(CLOCK_MONOTONIC, &current);
        } while ((current.tv_sec - start.tv_sec) + (current.tv_nsec - start.tv_nsec) / 1e9 < 1.0);

    }

    // printf("Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id, data->id);
    pthread_mutex_unlock(&logMutex);
    fprintf(logFile,"Parent ID : %d thread ID: %d, shutting itself off\n", data->parent_id, data->id);
    pthread_mutex_unlock(&logMutex);
    pthread_exit(NULL);

}

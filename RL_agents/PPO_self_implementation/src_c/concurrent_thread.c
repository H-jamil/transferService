#include "thread.h"
#include <string.h>
#include "hash_table.h"
extern int current_concurrency;
extern pthread_mutex_t concurrency_mutex;
extern int current_parallelism;
extern pthread_mutex_t parallelism_mutex;

extern FILE* logFile;
extern pthread_mutex_t logMutex;

extern HashTable *dataGenTable;

char* extract_filename(const char* url) {
    char *last_slash = strrchr(url, '/');
    if (last_slash) {
        return last_slash + 1;  // Move past the '/'
    }
    return (char*)url;  // Return the original URL if no slash found
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
    int active_parallel_value = get_parallel_value();
    adjust_parallel_workers(data->thread_data, active_parallel_value);
}

double get_file_size_from_url(const char *url) {
    CURL *curl;
    CURLcode res;
    double file_size = 0.0;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request only, for getting the file size
        curl_easy_setopt(curl, CURLOPT_FILETIME, 1L);

        res = curl_easy_perform(curl);
        if (CURLE_OK == res) {
            res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &file_size);
        }
        curl_easy_cleanup(curl);
    }
    return file_size;
}

Queue* get_generator_queue(Queue *files_need_to_be_downloaded, int chunk_size){
    Queue *generator_queue=queue_create();
    // printf("get_generator_queue: Queue size: %d\n",queue_size(files_need_to_be_downloaded));
    while(queue_size(files_need_to_be_downloaded)>0){
        char *file_url=queue_pop(files_need_to_be_downloaded);
        double size_of_file=get_file_size_from_url(file_url);
        DataGenerator *gen = data_generator_init(file_url, extract_filename(file_url), size_of_file,chunk_size);
        pthread_mutex_lock(&logMutex);
        fprintf(logFile, "Gen address: %p\n", gen);
        pthread_mutex_unlock(&logMutex);
        hash_table_add(dataGenTable, gen);
        queue_push(generator_queue,gen);
    }
    return generator_queue;
}

void* ConcurrencyThreadFunc(void* arg) {
    ConcurrencyWorkerData* data = (ConcurrencyWorkerData*) arg;
    int active_parallel_value = 0;
    int old_active_parallel_value = -1;
    pthread_t threads[MAX_PARALLELISM];
    ParallelWorkerData thread_data[MAX_PARALLELISM];
    DataGenerator *gen;
    if ((gen= queue_pop(data->files_need_to_be_downloaded)) == NULL) {
        pthread_mutex_lock(&logMutex);
        fprintf(logFile, "Concurrent Thread %d shutting down because there are no files to download\n", data->id);
        pthread_mutex_unlock(&logMutex);
        return NULL;
    }

    // 0x562a6a700000 : real
    // 0x555555500000 : gdb
    // uintptr_t myConst = (uintptr_t)0x555555500000;
    // uintptr_t myMask = (uintptr_t)0xFFFFFFF00000;
    // if (((uintptr_t) gen & myMask) != myConst) {
    //     printf("Gen address goofed: %p\n", gen);
    // }

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

    pthread_mutex_lock(&logMutex);
    fprintf(logFile, "Concurrent Thread %d creating all parallel threads (status: paused)\n", data->id);
    pthread_mutex_unlock(&logMutex);

    while(data->active) {
        pthread_mutex_lock(&concurrency_mutex);
        if (data->id >= current_concurrency) {
            queue_push(data->files_need_to_be_downloaded,gen);
            gen = NULL;
            pthread_mutex_unlock(&concurrency_mutex);
            pause_concurrency_worker(data);
            sleep(UPDATE_TIME);
            continue;
        }
        pthread_mutex_unlock(&concurrency_mutex);
        if (is_finished(gen)) {
            // queue_push(data->files_downloaded, gen); // Add this line to push the finished generator
            if ((gen= queue_pop(data->files_need_to_be_downloaded)) == NULL) {
                pthread_mutex_lock(&logMutex);
                fprintf(logFile, "Concurrent Thread %d shutting down because there are no files to download\n", data->id);
                pthread_mutex_unlock(&logMutex);
                for(int i = 0; i < MAX_PARALLELISM; i++) {
                    thread_data[i].active = 0;
                }
                return NULL;
            }

            for(int i = 0; i < MAX_PARALLELISM; i++) {
            thread_data[i].data_generator = gen;
            }
            data->thread_data = thread_data;
        }
        resume_concurrency_worker(data);
        active_parallel_value = get_parallel_value();
        if (active_parallel_value != old_active_parallel_value) {
            adjust_parallel_workers(thread_data, active_parallel_value);
            old_active_parallel_value = active_parallel_value;
        }
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

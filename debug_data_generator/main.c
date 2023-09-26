#include "parallel.h"
#include "concurrency.h"
#include "queue.h"
#include "data_generator.h"

#include <string.h> // for memset
#include <unistd.h> // for close()
#include <curl/curl.h>

#define CHUNK_SIZE 80000000
#define MAX_FILE_NUMBER 32

int current_concurrency;
pthread_mutex_t concurrency_mutex;
int current_parallelism;
pthread_mutex_t parallelism_mutex;
FILE* logFile;
pthread_mutex_t logMutex;

char* extract_filename(const char* url) {
    char *last_slash = strrchr(url, '/');
    if (last_slash) {
        return last_slash + 1;  // Move past the '/'
    }
    return (char*)url;  // Return the original URL if no slash found
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
    while(queue_size(files_need_to_be_downloaded)>0){
        char *file_url=queue_pop(files_need_to_be_downloaded);
        double size_of_file=get_file_size_from_url(file_url);
        DataGenerator *gen = data_generator_init(file_url, extract_filename(file_url), size_of_file,chunk_size);
        queue_push(generator_queue,gen);
    }
    return generator_queue;
}

Queue* get_generator_queue_with_data_chunks(Queue *files_need_to_be_downloaded, int chunk_size){
    Queue *generator_queue=queue_create();
    parallel_work_data *chunk;
    while(queue_size(files_need_to_be_downloaded)>0){
        char *file_url=queue_pop(files_need_to_be_downloaded);
        double size_of_file=get_file_size_from_url(file_url);
        DataGenerator *gen = data_generator_init(file_url, extract_filename(file_url), size_of_file,chunk_size);
        while(1){
            if((chunk=data_generator_next(gen))!=NULL){
            queue_push(generator_queue,chunk);
            }
            else{
                break;
            }
        }
    }
    return generator_queue;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {  // Update argument check to 3 (program name, IP address, log file)
        fprintf(stderr, "Usage: %s <IP_ADDRESS>\n", argv[0]);
        return 1;
    }
    // Initialize concurrency
    char *log_filename = "log.txt";
    char *ip_address = argv[1];
    logFile = fopen(log_filename, "w");
        if(!logFile) {
            perror("Error opening log file");
            exit(1);
        }
    current_concurrency = 0;
    current_parallelism = 0;
    pthread_mutex_init(&concurrency_mutex, NULL);
    pthread_mutex_init(&parallelism_mutex, NULL);
    pthread_mutex_init(&logMutex, NULL);

    curl_global_init(CURL_GLOBAL_DEFAULT);


    // Initialize queue

    Queue *files_need_to_be_downloaded=queue_create();
    Queue *files_downloaded=queue_create();
    for (int i = 0; i < MAX_FILE_NUMBER; i++) {
        char file_url[100];
        sprintf(file_url, "http://%s/FILE%d", ip_address, i);
        queue_push(files_need_to_be_downloaded, strdup(file_url));
        queue_push(files_downloaded, strdup(file_url));
    }
    Queue *generator_queue=get_generator_queue(files_need_to_be_downloaded,CHUNK_SIZE);
    Queue *generator_queue_with_data_chunks=get_generator_queue_with_data_chunks(files_downloaded,CHUNK_SIZE);

    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;  // All threads start in paused state
        thread_data[i].thread_data=NULL;
        thread_data[i].files_need_to_be_downloaded=generator_queue;
        thread_data[i].files_downloaded=generator_queue_with_data_chunks;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }
int count=0;

// while(job_done==0){

while(count<2){
        pthread_mutex_lock(&logMutex);
        fprintf(logFile,"concurrent threads 2 parallel threads 3\n");
        pthread_mutex_unlock(&logMutex);
        // printf("concurrent threads 2 parallel threads 3\n");
        set_parallel_value(3);
        set_concurrent_value(2);
        sleep(2*UPDATE_TIME);

        pthread_mutex_lock(&logMutex);
        fprintf(logFile,"concurrent threads 1 parallel threads 1\n");
        pthread_mutex_unlock(&logMutex);
        // printf("concurrent threads 1 parallel threads 1\n");
        set_parallel_value(1);
        set_concurrent_value(1);
        sleep(2*UPDATE_TIME);

        pthread_mutex_lock(&logMutex);
        fprintf(logFile,"concurrent threads 2 parallel threads 2\n");
        pthread_mutex_unlock(&logMutex);
        // printf("concurrent threads 2 parallel threads 2\n");
        set_parallel_value(2);
        set_concurrent_value(2);
        sleep(2*UPDATE_TIME);

        pthread_mutex_lock(&logMutex);
        fprintf(logFile,"concurrent threads 3 parallel threads 3\n");
        pthread_mutex_unlock(&logMutex);
        // printf("concurrent threads 3 parallel threads 3\n");
        set_parallel_value(3);
        set_concurrent_value(3);
        sleep(2*UPDATE_TIME);

        pthread_mutex_lock(&logMutex);
        fprintf(logFile,"concurrent threads 16 parallel threads 16\n");
        pthread_mutex_unlock(&logMutex);
        // printf("concurrent threads 4 parallel threads 4\n");
        set_parallel_value(16);
        set_concurrent_value(16);
        sleep(2*UPDATE_TIME);

        count++;
   }
    pthread_mutex_lock(&logMutex);
    fprintf(logFile,"concurrent threads MAX parallel threads MAX\n");
    pthread_mutex_unlock(&logMutex);
    // printf("concurrent threads MAX parallel threads MAX\n");
    set_parallel_value(MAX_PARALLELISM);
    set_concurrent_value(MAX_CONCURRENCY);
    sleep(UPDATE_TIME);



    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].active = 0;
    }

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
    }
    pthread_mutex_destroy(&concurrency_mutex);
    pthread_mutex_destroy(&parallelism_mutex);
    pthread_mutex_destroy(&logMutex);
    queue_destroy(files_need_to_be_downloaded);
    queue_destroy(files_downloaded);
    queue_destroy(generator_queue);
    queue_destroy(generator_queue_with_data_chunks);
    curl_global_cleanup();
    while(queue_size(generator_queue_with_data_chunks)>0){
        parallel_work_data *chunk=queue_pop(generator_queue_with_data_chunks);
        free(chunk->url);
        free(chunk->output_file);
        free(chunk);
    }
    while(queue_size(generator_queue)>0){
        DataGenerator *gen=queue_pop(generator_queue);
        data_generator_free(gen);
    }

    fclose(logFile);
    return 0;
}


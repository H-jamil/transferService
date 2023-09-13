#include "parallel.h"
#include "concurrency.h"
#include "queue.h"
#include <curl/curl.h>


#define MAX_FILE_NUMBER 8
#define CHUNK_SIZE 5000000

int current_concurrency;
pthread_mutex_t concurrency_mutex;
int current_parallelism;
pthread_mutex_t parallelism_mutex;
int number_of_files_downloaded;
pthread_mutex_t number_of_files_downloaded_mutex;
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
        queue_push(generator_queue, gen);
    }
    return generator_queue;
}

void print_queue_data_generator_ids(Queue* queue) {
    pthread_mutex_lock(&(queue->lock));

    Node* current = queue->front;
    printf("DataGenerator IDs in Queue:\n");
    while (current != NULL) {
        DataGenerator* gen = (DataGenerator*) current->data;
        if (gen->url) {
            printf("%s\n", gen->url);
        }
        current = current->next;
    }

    pthread_mutex_unlock(&(queue->lock));
}

int main(int argc, char** argv){
  current_concurrency = 0;
  current_parallelism = 0;
  number_of_files_downloaded = 0;
  pthread_mutex_init(&concurrency_mutex, NULL);
  pthread_mutex_init(&parallelism_mutex, NULL);
  pthread_mutex_init(&logMutex, NULL);
  pthread_mutex_init(&number_of_files_downloaded_mutex, NULL);
  logFile = fopen("output.log", "w");
    if(!logFile) {
        perror("Error opening log file");
        exit(1);
    }
    curl_global_init(CURL_GLOBAL_DEFAULT);
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    Queue *file_need_to_be_downloaded=queue_create();
    for (int i = 0; i < MAX_FILE_NUMBER; i++) {
        char file_url[100];
        sprintf(file_url, "http://128.205.218.120/FILE%d", i);
        queue_push(file_need_to_be_downloaded, strdup(file_url));
    }
    Queue *generator_queue=get_generator_queue(file_need_to_be_downloaded,CHUNK_SIZE);
    pthread_mutex_lock(&logMutex);
    fprintf(logFile, "From Main thread beginning Queue size: %d\n", queue_size(generator_queue));
    pthread_mutex_unlock(&logMutex);
    print_queue_data_generator_ids(generator_queue);

    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;  // All threads start in paused state
        thread_data[i].thread_data=NULL;
        thread_data[i].files_need_to_be_downloaded=generator_queue;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }

    sleep(1);
    print_queue_data_generator_ids(generator_queue);
    while(number_of_files_downloaded<MAX_FILE_NUMBER){

      pthread_mutex_lock(&logMutex);
      fprintf(logFile, "MT changed P to 2 and C to 2 with %d files downloaded\n",number_of_files_downloaded);
      pthread_mutex_unlock(&logMutex);
      set_parallel_value(2);
      set_concurrent_value(2);
      sleep(2*UPDATE_TIME);

      // pthread_mutex_lock(&logMutex);
      // fprintf(logFile, "MT changed P to 1 and C to 1 with %d files downloaded\n",number_of_files_downloaded);
      // pthread_mutex_unlock(&logMutex);
      // set_parallel_value(1);
      // set_concurrent_value(1);
      // sleep(2*UPDATE_TIME);

      // pthread_mutex_lock(&logMutex);
      // fprintf(logFile, "MT changed P to 3 and C to 3 with %d files downloaded\n",number_of_files_downloaded);
      // pthread_mutex_unlock(&logMutex);
      // set_parallel_value(3);
      // set_concurrent_value(3);
      // sleep(2*UPDATE_TIME);

      // pthread_mutex_lock(&logMutex);
      // fprintf(logFile, "MT changed P to 4 and C to 4 with %d files downloaded\n",number_of_files_downloaded);
      // pthread_mutex_unlock(&logMutex);
      // set_parallel_value(4);
      // set_concurrent_value(4);
      // sleep(2*UPDATE_TIME);

      // pthread_mutex_lock(&logMutex);
      // fprintf(logFile, "MT changed P to 3 and C to 3 with %d files downloaded\n",number_of_files_downloaded);
      // pthread_mutex_unlock(&logMutex);
      // set_parallel_value(3);
      // set_concurrent_value(3);
      // sleep(2*UPDATE_TIME);
    }
    printf("From Main thread ending Queue size: %d\n",queue_size(generator_queue));

    // pthread_mutex_lock(&logMutex);
    // fprintf(logFile, "MT changed P to MAX and C to MAX with %d files downloaded\n",number_of_files_downloaded);
    // pthread_mutex_unlock(&logMutex);
    // set_parallel_value(MAX_PARALLELISM);
    // set_concurrent_value(MAX_CONCURRENCY);
    // sleep(2*UPDATE_TIME);

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].active = 0;
    }

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
    }
    curl_global_cleanup();
    queue_destroy(file_need_to_be_downloaded);
    queue_destroy(generator_queue);
    pthread_mutex_destroy(&concurrency_mutex);
    pthread_mutex_destroy(&parallelism_mutex);
    pthread_mutex_destroy(&logMutex);
    pthread_mutex_destroy(&number_of_files_downloaded_mutex);
    fclose(logFile);
    printf("All done!\n");
    return 0;

}

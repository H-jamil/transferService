#include "parallel.h"
#include "concurrency.h"
#include <curl/curl.h>

#define MAX_FILE_NUMBER 4
#define CHUNK_SIZE 5000000
int current_concurrency;
pthread_mutex_t concurrency_mutex;

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


int main(int argc, char** argv) {
    // Initialize concurrency
    current_concurrency = 0;
    pthread_mutex_init(&concurrency_mutex, NULL);
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    DataGenerator* gen[MAX_FILE_NUMBER];
    curl_global_init(CURL_GLOBAL_DEFAULT);

    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;  // All threads start in paused state
        thread_data[i].thread_data=NULL;
        char file_url[100];
        sprintf(file_url, "http://128.205.218.120/FILE%d", i);
        double size_of_file = get_file_size_from_url(file_url);
        gen[i] = data_generator_init(file_url, extract_filename(file_url), size_of_file,CHUNK_SIZE);
        thread_data[i].generator = gen[i];
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_mutex_init(&thread_data[i].parallel_value_mutex, NULL);
        // Create threads
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }
    // Adjust threads as described
    sleep(1);  // Give threads a moment to start

    printf("concurrent threads 2 parallel threads 3\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 3);
        }
    set_concurrent_value(2);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 1 parallel threads 2\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 2);
        }
    set_concurrent_value(1);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 4 parallel threads 4\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 4);
        }
    set_concurrent_value(4);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 1 parallel threads 1\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 1);
        }
    set_concurrent_value(1);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 2 parallel threads 2\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 2);
        }
    set_concurrent_value(2);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 3 parallel threads 3\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 3);
        }
    set_concurrent_value(3);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 4 parallel threads 4\n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], 4);
        }
    set_concurrent_value(4);
    sleep(4*UPDATE_TIME);


    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].active = 0;
    }

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
        pthread_mutex_destroy(&thread_data[i].parallel_value_mutex);
    }
    for(int i = 0; i < MAX_FILE_NUMBER; i++) {
        data_generator_free(gen[i]);
    }
    curl_global_cleanup();
    pthread_mutex_destroy(&concurrency_mutex);
    return 0;
}


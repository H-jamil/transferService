#include "thread.h"
#include "data_generator.h"
#include "queue.h"
#include <string.h>
int current_concurrency = MAX_CONCURRENCY;
pthread_mutex_t concurrency_mutex;
FILE* logFile;
pthread_mutex_t logMutex;
#define CHUNK_SIZE 10000000
#define MAX_FILE_NUMBER 8


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

int main() {
    int user_parallelism, user_concurrency, queue_size_num;
    char continueInput = 'y';
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    pthread_mutex_init(&concurrency_mutex, NULL);
    // Initialize the log mutex
    pthread_mutex_init(&logMutex, NULL);
    logFile = fopen("output.log", "w");
    if(!logFile) {
        perror("Error opening log file");
        exit(1);
    }
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Creating Queue
    Queue *files_need_to_be_downloaded=queue_create();
    Queue *files_downloaded=queue_create();
    for (int i = 0; i < MAX_FILE_NUMBER; i++) {
        char file_url[100];
        sprintf(file_url, "http://128.205.218.120/FILE%d", i);
        queue_push(files_need_to_be_downloaded, strdup(file_url));
    }
    Queue *generator_queue=get_generator_queue(files_need_to_be_downloaded,CHUNK_SIZE);

    // Initialize and start concurrent threads
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        thread_data[i].parallel_value = 0;
        thread_data[i].files_need_to_be_downloaded=generator_queue;
        thread_data[i].files_downloaded=files_downloaded;
        thread_data[i].chunk_size=CHUNK_SIZE;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_mutex_init(&thread_data[i].parallel_value_mutex, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }
    do {
        // Prompt user to enter values
        printf("Enter parallelism value (1 to MAX_PARALLELISM): ");
        scanf("%d", &user_parallelism);

        printf("Enter concurrency value (1 to MAX_CONCURRENCY): ");
        scanf("%d", &user_concurrency);

        pthread_mutex_lock(&logMutex);
        fprintf(logFile, "Main Thread changed parallelism to %d and concurrency to %d\n", user_parallelism, user_concurrency);
        pthread_mutex_unlock(&logMutex);

        for (int i = 0; i < MAX_CONCURRENCY; i++) {
            set_parallel_value(&thread_data[i], user_parallelism);
        }
        set_concurrent_value(user_concurrency);
        sleep(2*UPDATE_TIME);

        // Ask user if they want to continue changing values
        queue_size_num = queue_size(files_downloaded);
        printf("Downloaded Queue Size: %d\n", queue_size_num);
        printf("Do you want to continue changing values? (y/n): ");
        scanf(" %c", &continueInput); // Note the space before %c to consume any leftover '\n' from the previous input

        if (queue_size_num>= MAX_FILE_NUMBER){
            break;
        }

    } while(continueInput != 'n');
    print_queue_data_generator_ids(files_downloaded);
    // fprintf(logFile,"Main Thread change parallelism to MAX and concurrency to MAX\n");
    // pthread_mutex_unlock(&logMutex);
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], MAX_PARALLELISM);
    }
    set_concurrent_value(MAX_CONCURRENCY);
    sleep(2*UPDATE_TIME);
    // pthread_mutex_lock(&logMutex);
    // printf("Main thread is shutting off all concurrent threads\n");
    // fprintf(logFile,"Main thread is shutting off all concurrent threads\n");
    // pthread_mutex_unlock(&logMutex);
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        printf("Main thread is shutting off concurrent thread %d\n", i);
        thread_data[i].active = 0;
    }

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
        pthread_mutex_destroy(&thread_data[i].parallel_value_mutex);
    }

    printf("All done\n");
    // pthread_mutex_lock(&logMutex);
    // fprintf(logFile,"All done\n");
    // pthread_mutex_unlock(&logMutex);
    fclose(logFile);
    pthread_mutex_destroy(&concurrency_mutex);
    pthread_mutex_destroy(&logMutex);
    curl_global_cleanup();
    queue_destroy(files_need_to_be_downloaded);
    queue_destroy(files_downloaded);
    queue_destroy(generator_queue);
    return 0;
}

#include "thread.h"
int current_concurrency = MAX_CONCURRENCY;
pthread_mutex_t concurrency_mutex;
FILE* logFile;
pthread_mutex_t logMutex;

int main() {
    int user_parallelism, user_concurrency;
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
    // Initialize and start concurrent threads
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        thread_data[i].parallel_value = 0;
        thread_data[i].file_name = malloc(50 * sizeof(char));  // Assuming a maximum length of 50 characters for file_name
        sprintf(thread_data[i].file_name, "http://128.205.218.120/file1GB_%d", i);
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_mutex_init(&thread_data[i].parallel_value_mutex, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }

    // pthread_mutex_lock(&logMutex);
    // // printf("Main Thread change parallelism to 3 and concurrency to 1\n");
    // fprintf(logFile,"Main Thread change parallelism to 3 and concurrency to 1\n");
    // pthread_mutex_unlock(&logMutex);
    // for (int i = 0; i < MAX_CONCURRENCY; i++) {
    //     set_parallel_value(&thread_data[i], 3);
    // }
    // set_concurrent_value(1);
    // sleep(2*UPDATE_TIME);
    // pthread_mutex_lock(&logMutex);
    // fprintf(logFile,"Main Thread change parallelism to 2 and concurrency to 2\n");
    // // printf("Main Thread change parallelism to 2 and concurrency to 2\n");
    // pthread_mutex_unlock(&logMutex);
    // for (int i = 0; i < MAX_CONCURRENCY; i++) {
    //     set_parallel_value(&thread_data[i], 2);
    // }
    // set_concurrent_value(2);
    // sleep(2*UPDATE_TIME);
    // pthread_mutex_lock(&logMutex);
    // fprintf(logFile,"Main Thread change parallelism to 1 and concurrency to 1\n");
    // // printf("Main Thread change parallelism to 1 and concurrency to 1\n");
    // pthread_mutex_unlock(&logMutex);
    // for (int i = 0; i < MAX_CONCURRENCY; i++) {
    //     set_parallel_value(&thread_data[i], 1);
    // }
    // set_concurrent_value(1);
    // sleep(2*UPDATE_TIME);
    // pthread_mutex_lock(&logMutex);

    // printf("Main Thread change parallelism to MAX and concurrency to 4\n");
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
        printf("Do you want to continue changing values? (y/n): ");
        scanf(" %c", &continueInput); // Note the space before %c to consume any leftover '\n' from the previous input

    } while(continueInput != 'n');

    fprintf(logFile,"Main Thread change parallelism to MAX and concurrency to MAX\n");
    pthread_mutex_unlock(&logMutex);
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], MAX_PARALLELISM);
    }
    set_concurrent_value(MAX_CONCURRENCY);
    sleep(2*UPDATE_TIME);

    pthread_mutex_lock(&logMutex);
    // printf("Main thread is shutting off all concurrent threads\n");
    fprintf(logFile,"Main thread is shutting off all concurrent threads\n");
    pthread_mutex_unlock(&logMutex);
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].active = 0;
    }

    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        pthread_join(threads[i], NULL);
        pthread_mutex_destroy(&thread_data[i].pause_mutex);
        pthread_cond_destroy(&thread_data[i].pause_cond);
        pthread_mutex_destroy(&thread_data[i].parallel_value_mutex);
    }
    pthread_mutex_destroy(&concurrency_mutex);
    // printf("All done\n");
    pthread_mutex_lock(&logMutex);
    fprintf(logFile,"All done\n");
    pthread_mutex_unlock(&logMutex);
    fclose(logFile);
    pthread_mutex_destroy(&logMutex);

    return 0;
}

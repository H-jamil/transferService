#include "thread.h"
int current_concurrency = MAX_CONCURRENCY;
pthread_mutex_t concurrency_mutex;

int main() {
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    pthread_mutex_init(&concurrency_mutex, NULL);

    // Initialize and start concurrent threads
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        thread_data[i].parallel_value = 0;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_mutex_init(&thread_data[i].parallel_value_mutex, NULL);
        pthread_mutex_init(&thread_data[i].global_concurrency_value_mutex, NULL);
        thread_data[i].global_concurrency_value = MAX_CONCURRENCY;  // Set initial value
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }

    printf("Main Thread change parallelism to Max \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], MAX_PARALLELISM);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 2 \n");
    set_concurrent_value(&thread_data[0], 2); // Using the first thread data structure to modify the shared concurrency value
    // for (int i = 0; i < 2; i++) {
    //     pause_concurrency_worker(&thread_data[i]);
    // }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change parallelism to 1 \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], 1);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 4 \n");
    // for (int i = 0; i < 2; i++) {
    //     resume_concurrency_worker(&thread_data[i]);
    // }
    set_concurrent_value(&thread_data[0], 4); // Using the first thread data structure to modify the shared concurrency value
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 3 \n");
    // for (int i = 0; i < 1; i++) {
    //     pause_concurrency_worker(&thread_data[i]);
    // }
    set_concurrent_value(&thread_data[0], 3); // Using the first thread data structure to modify the shared concurrency value
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 1 \n");
    // for (int i = 1; i < 3; i++) {
    //     pause_concurrency_worker(&thread_data[i]);
    // }
    set_concurrent_value(&thread_data[0], 1); // Using the first thread data structure to modify the shared concurrency value
    sleep(2*UPDATE_TIME);

    printf("Main Thread change parallelism to 3 \n");
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        set_parallel_value(&thread_data[i], 3);
    }
    sleep(2*UPDATE_TIME);

    printf("Main Thread change concurrency to 4 \n");
    // for (int i = 0; i < 3; i++) {
    //     resume_concurrency_worker(&thread_data[i]);
    // }
    set_concurrent_value(&thread_data[0], 4); // Using the first thread data structure to modify the shared concurrency value
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
    pthread_mutex_destroy(&concurrency_mutex);
    printf("All done\n");
    return 0;
}

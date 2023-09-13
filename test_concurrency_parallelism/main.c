#include "parallel.h"
#include "concurrency.h"

int current_concurrency;
pthread_mutex_t concurrency_mutex;
int current_parallelism;
pthread_mutex_t parallelism_mutex;

int main(int argc, char** argv) {
    // Initialize concurrency
    current_concurrency = 0;
    current_parallelism = 0;
    pthread_mutex_init(&concurrency_mutex, NULL);
    pthread_mutex_init(&parallelism_mutex, NULL);
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    for (int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 1;  // All threads start in paused state
        thread_data[i].thread_data=NULL;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }
int count=0;
// while(job_done==0){

while(count<5){

    printf("concurrent threads 2 parallel threads 3\n");
    set_parallel_value(3);
    set_concurrent_value(2);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 1 parallel threads 2\n");
    set_parallel_value(2);
    set_concurrent_value(1);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 4 parallel threads 4\n");
    set_parallel_value(4);
    set_concurrent_value(4);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 1 parallel threads 1\n");
    set_parallel_value(1);
    set_concurrent_value(1);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 2 parallel threads 2\n");
    set_parallel_value(2);
    set_concurrent_value(2);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 3 parallel threads 3\n");
    set_parallel_value(3);
    set_concurrent_value(3);
    sleep(2*UPDATE_TIME);

    printf("concurrent threads 4 parallel threads 4\n");
    set_parallel_value(4);
    set_concurrent_value(4);
    sleep(2*UPDATE_TIME);

    count++;
   }
    printf("concurrent threads MAX parallel threads MAX\n");
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
    return 0;
}


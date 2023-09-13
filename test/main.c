#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_THREADS 10
#define UPDATE_INTERVAL 3

typedef struct {
    int thread_num;
    pthread_t tid;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int paused;
    int parent_id;
    int terminate;
} ThreadInfo;

typedef struct {
    int thread_num;
    pthread_t tid;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int paused;
    int terminate;
    ThreadInfo* thread_info;
} concurrent_thread_info;

int activee_parallel_threads = MAX_THREADS;
int activee_concurrent_threads = MAX_THREADS;
pthread_mutex_t activee_concurrent_threads_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t activee_parallel_threads_lock = PTHREAD_MUTEX_INITIALIZER;

void pause_parallel_thread(ThreadInfo* info);
void resume_parallel_thread(ThreadInfo* info);
void terminate_parallel_thread(ThreadInfo* info);
void pause_concurrent_thread(concurrent_thread_info* info);
void resume_concurrent_thread(concurrent_thread_info* info);
void terminate_concurrent_thread(concurrent_thread_info* info);
void set_active_parallel_threads(int count, ThreadInfo thread_info[MAX_THREADS]);
void set_active_concurrent_threads(int count, concurrent_thread_info thread_info[MAX_THREADS]);
int get_active_parallel_threads();
int get_active_concurrent_threads();
void setup_active_parallel_threads(int count);
void setup_active_concurrent_threads(int count);


void pause_parallel_thread(ThreadInfo* info) {
    pthread_mutex_lock(&info->lock);
    info->paused = 1;
    pthread_mutex_unlock(&info->lock);
}

void resume_parallel_thread(ThreadInfo* info) {
    pthread_mutex_lock(&info->lock);
    if (info->paused) {
        info->paused = 0;
        pthread_cond_signal(&info->cond);
    }
    pthread_mutex_unlock(&info->lock);
}

void terminate_parallel_thread(ThreadInfo* info) {
    pthread_mutex_lock(&info->lock);
    info->terminate = 1;
    pthread_cond_signal(&info->cond);
    pthread_mutex_unlock(&info->lock);
    pthread_join(info->tid, NULL);
}

void pause_concurrent_thread(concurrent_thread_info* info) {
    pthread_mutex_lock(&info->lock);
    info->paused = 1;
    pthread_mutex_unlock(&info->lock);
}

void resume_concurrent_thread(concurrent_thread_info* info) {
    pthread_mutex_lock(&info->lock);
    if (info->paused) {
        info->paused = 0;
        pthread_cond_signal(&info->cond);
    }
    pthread_mutex_unlock(&info->lock);
}

void terminate_concurrent_thread(concurrent_thread_info* info) {
    pthread_mutex_lock(&info->lock);
    info->terminate = 1;
    pthread_cond_signal(&info->cond);
    pthread_mutex_unlock(&info->lock);
    pthread_join(info->tid, NULL);
}


void setup_active_parallel_threads(int count){
    pthread_mutex_lock(&activee_parallel_threads_lock);
    activee_parallel_threads = count;
    pthread_mutex_unlock(&activee_parallel_threads_lock);
}

void setup_active_concurrent_threads(int count){
    pthread_mutex_lock(&activee_concurrent_threads_lock);
    activee_concurrent_threads = count;
    pthread_mutex_unlock(&activee_concurrent_threads_lock);
}

void set_active_concurrent_threads(int count, concurrent_thread_info thread_info[MAX_THREADS]) {
    if (count > MAX_THREADS)
        count = MAX_THREADS;
    for (int i = 0; i < MAX_THREADS; i++) {
        if (i < count) {
            resume_concurrent_thread(&thread_info[i]);
        } else {
            pause_concurrent_thread(&thread_info[i]);
        }
    }
    setup_active_concurrent_threads(count);
}

void set_active_parallel_threads(int count, ThreadInfo thread_info[MAX_THREADS]) {
    if (count > MAX_THREADS)
        count = MAX_THREADS;
    for (int i = 0; i < MAX_THREADS; i++) {
        if (i < count) {
            resume_parallel_thread(&thread_info[i]);
        } else {
            pause_parallel_thread(&thread_info[i]);
        }
    }
    setup_active_parallel_threads(count);
}

void* parallel_function(void* arg) {
    ThreadInfo* info = (ThreadInfo*) arg;
    while (1) {
        pthread_mutex_lock(&info->lock);
        if (info->terminate) {
            pthread_mutex_unlock(&info->lock);
            pthread_exit(NULL);
        }
        while (info->paused && !info->terminate) {
            pthread_cond_wait(&info->cond, &info->lock);
        }
        pthread_mutex_unlock(&info->lock);
        printf("Parent ID: %d Thread ID: %d\n", info->parent_id, info->thread_num);
        sleep(1);
    }
    return NULL;
}

void* concurrent_function(void* arg) {
    int parallel_threads = 0;
    concurrent_thread_info* info = (concurrent_thread_info*) arg;

    ThreadInfo* thread_info_array = malloc(MAX_THREADS * sizeof(ThreadInfo));
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_info_array[i].thread_num = i;
        pthread_mutex_init(&thread_info_array[i].lock, NULL);
        pthread_cond_init(&thread_info_array[i].cond, NULL);
        thread_info_array[i].paused = 1;
        thread_info_array[i].parent_id = info->thread_num;
        thread_info_array[i].terminate = 0;
        pthread_create(&thread_info_array[i].tid, NULL, parallel_function, &thread_info_array[i]);
    }
    info->thread_info = thread_info_array;

    while (1) {
        pthread_mutex_lock(&info->lock);
        if (info->terminate) {
            pthread_mutex_unlock(&info->lock);
            for (int i = 0; i < MAX_THREADS; i++) {
                terminate_parallel_thread(&thread_info_array[i]);
            }
            free(thread_info_array);
            pthread_exit(NULL);
        }
        while (info->paused && !info->terminate) {
            pthread_cond_wait(&info->cond, &info->lock);
        }
        pthread_mutex_unlock(&info->lock);
        sleep(UPDATE_INTERVAL);
    }
    return NULL;
}

int main() {
    pthread_mutex_init(&activee_parallel_threads_lock, NULL);
    pthread_mutex_init(&activee_concurrent_threads_lock, NULL);
    concurrent_thread_info thread_info[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; i++) {
        thread_info[i].thread_num = i;
        pthread_mutex_init(&thread_info[i].lock, NULL);
        pthread_cond_init(&thread_info[i].cond, NULL);
        thread_info[i].paused = 0;
        thread_info[i].terminate = 0;
        pthread_create(&thread_info[i].tid, NULL, concurrent_function, &thread_info[i]);
    }

    printf("concurrency 2 and parallelism 4\n");
    setup_active_parallel_threads(4);
    set_active_concurrent_threads(2,thread_info);
    sleep(5);
    printf("concurrency 3 and parallelism 2\n");
    setup_active_parallel_threads(2);
    set_active_concurrent_threads(3,thread_info);
    sleep(5);
    printf("Terminating all threads\n");
    setup_active_parallel_threads(MAX_THREADS);
    set_active_concurrent_threads(MAX_THREADS,thread_info);
    sleep(5);
    for (int i = 0; i < MAX_THREADS; i++) {
        terminate_concurrent_thread(&thread_info[i]);
    }
    pthread_mutex_destroy(&activee_concurrent_threads_lock);
    pthread_mutex_destroy(&activee_parallel_threads_lock);
    exit(0);
}

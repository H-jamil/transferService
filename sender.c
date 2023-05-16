#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "/home/jamilm/libzmq/include/zmq.h"
#include <assert.h>

#define CHUNK_SIZE 1048576 // 1 MB
#define FILE_PATH "/home/jamilm/transferService/files_to_send/FILE0"

typedef struct {
    void *socket;
    long start;
    long end;
} ThreadData;

typedef struct {
    size_t count;
    ThreadData** data;
    size_t capacity;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} Queue;

Queue queue = {
    .count = 0,
    .data = NULL,
    .capacity = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER
};

void enqueue(ThreadData* data) {
    pthread_mutex_lock(&queue.mutex);
    while (queue.count == queue.capacity) {
        pthread_cond_wait(&queue.cond, &queue.mutex);
    }
    queue.data[queue.count++] = data;
    pthread_cond_signal(&queue.cond);
    pthread_mutex_unlock(&queue.mutex);
}

ThreadData* dequeue() {
    pthread_mutex_lock(&queue.mutex);
    while (queue.count == 0) {
        pthread_cond_wait(&queue.cond, &queue.mutex);
    }
    ThreadData* data = queue.data[--queue.count];
    pthread_cond_signal(&queue.cond);
    pthread_mutex_unlock(&queue.mutex);
    return data;
}

void* worker_thread(void* arg) {
    while (1) {
        // get next chunk from queue
        ThreadData* data = dequeue();

        // open file
        FILE* file = fopen(FILE_PATH, "rb");
        if (file == NULL) {
            fprintf(stderr, "Failed to open file\n");
            return NULL;
        }

        // seek to start of chunk
        if (fseek(file, data->start, SEEK_SET) != 0) {
            fprintf(stderr, "Failed to seek file\n");
            fclose(file);
            return NULL;
        }

        // read chunk
        size_t size = data->end - data->start;
        char* buffer = malloc(size);
        if (fread(buffer, 1, size, file) != size) {
            fprintf(stderr, "Failed to read file\n");
            fclose(file);
            free(buffer);
            return NULL;
        }

        // close file
        fclose(file);

        // send chunk over socket
        if (zmq_send(data->socket, buffer, size, 0) < 0) {
            fprintf(stderr, "Failed to send data\n");
        }

        // clean up
        free(buffer);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <IP> <port> <num_threads>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *ip = argv[1];
    int port = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    char endpoint[128];
    sprintf(endpoint, "tcp://%s:%d", ip, port);
    if (num_threads <= 0) {
        fprintf(stderr, "NUM_THREADS must be a positive integer\n");
        return 1;
    }

    // get size of file
    FILE* file = fopen(FILE_PATH, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file\n");
        return 1;
    }
    fseek(file, 0L, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

        // create ZeroMQ context
    void* context = zmq_ctx_new();

    // create worker threads
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, NULL);
    }

    // initialize queue
    queue.data = malloc(num_threads * sizeof(ThreadData*));
    queue.capacity = num_threads;

    // create tasks
    ThreadData* tasks = malloc(num_threads * sizeof(ThreadData));
    for (int i = 0; i < num_threads; i++) {
        // calculate start and end of chunk
        long start = (file_size / num_threads) * i;
        long end = (i == num_threads - 1) ? file_size : (file_size / num_threads) * (i + 1);

        // create ZeroMQ socket
        void* socket = zmq_socket(context, ZMQ_PUSH);
        zmq_connect(socket, endpoint);

        // create task data
        tasks[i].socket = socket;
        tasks[i].start = start;
        tasks[i].end = end;

        // enqueue task
        enqueue(&tasks[i]);
    }

    // wait for worker threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    // send end message
    void* socket = zmq_socket(context, ZMQ_PUSH);
    zmq_connect(socket, endpoint);
    zmq_send(socket, "end", 3, 0); // "end" string and null terminator

    // clean up
    free(threads);
    free(tasks);
    free(queue.data);
    zmq_ctx_destroy(context);

    return 0;
}


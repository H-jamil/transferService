#include "thread.h"
#include "data_generator.h"
#include "queue.h"
#include <stdbool.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h> // for memset
#include <unistd.h> // for close()

int current_concurrency = 0;
pthread_mutex_t concurrency_mutex;
FILE* logFile;
pthread_mutex_t logMutex;
int current_parallelism = 0;
pthread_mutex_t parallelism_mutex;

#define CHUNK_SIZE 5000000
#define MAX_FILE_NUMBER 8
#define PORT 8080

atomic_int downloaded_chunks;


typedef struct monitor_args {
    atomic_int *downloaded_chunks;
    FILE *logFile;
    pthread_mutex_t *logMutex;
    int total_chunks;
    atomic_int job_done;
    int chunk_size;
} monitor_args;

void* monitor_thread(void* arg) {
    monitor_args *args = (monitor_args*) arg;
    unsigned long long old_bytes_downloaded = 0;
    unsigned long long new_bytes_downloaded = 0;
    int current_downloaded_chunks=0;
    while (1) {
        sleep(1);
        current_downloaded_chunks = atomic_load(args->downloaded_chunks);
        new_bytes_downloaded = (unsigned long long)current_downloaded_chunks * args->chunk_size;
        double throughput = (double)(new_bytes_downloaded - old_bytes_downloaded) * 8 / 1000000000;
        pthread_mutex_lock(args->logMutex);
        fprintf(args->logFile, "Monitor Thread Throughput : %.2f Gbps\n", throughput);
        pthread_mutex_unlock(args->logMutex);
        if (current_downloaded_chunks == args->total_chunks) {
            atomic_store(&args->job_done, 1);
            break;
        }
        old_bytes_downloaded = new_bytes_downloaded;
    }
    return NULL;
}

Queue* get_generator_queue_with_data_chunks(Queue *files_need_to_be_downloaded, int chunk_size){
    Queue *generator_queue=queue_create();
    parallel_work_data *chunk;
    while(queue_size(files_need_to_be_downloaded)>0){
        char *file_url=queue_pop(files_need_to_be_downloaded);
        // printf("file_url: %s\n",file_url);
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

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <IP_ADDRESS>\n", argv[0]);
        return 1;
    }

    char termination_msg[] = "TERMINATE\0";
    char ok_msg[] = "OK\0";
    char *ip_address = argv[1];
    int user_parallelism, user_concurrency;
    // char continueInput = 'y';
    atomic_init(&downloaded_chunks, 0); // Initialize size to 0
    pthread_t threads[MAX_CONCURRENCY];
    ConcurrencyWorkerData thread_data[MAX_CONCURRENCY];
    pthread_mutex_init(&concurrency_mutex, NULL);
    pthread_mutex_init(&parallelism_mutex, NULL);
    pthread_mutex_init(&logMutex, NULL);
    logFile = fopen("output.log", "w");
    if(!logFile) {
        perror("Error opening log file");
        exit(1);
    }
    curl_global_init(CURL_GLOBAL_DEFAULT);


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

    /*
    //server initialization for python client to connect
    */
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("Setsockopt error");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }


    monitor_args args;
    pthread_t monitor_tid;
    args.downloaded_chunks = &downloaded_chunks;
    args.logFile = logFile;
    args.logMutex = &logMutex;
    args.total_chunks = queue_size(generator_queue_with_data_chunks);
    atomic_init(&args.job_done, 0);
    args.chunk_size = CHUNK_SIZE;
    pthread_create(&monitor_tid, NULL, monitor_thread, &args);

    printf("From Main : Total %d chunks will be downloaded\n", queue_size(generator_queue_with_data_chunks));
    for(int i = 0; i < MAX_CONCURRENCY; i++) {
        thread_data[i].id = i;
        thread_data[i].active = 1;
        thread_data[i].paused = 0;
        thread_data[i].files_need_to_be_downloaded=generator_queue;
        thread_data[i].files_downloaded=files_downloaded;
        thread_data[i].chunk_size=CHUNK_SIZE;
        pthread_mutex_init(&thread_data[i].pause_mutex, NULL);
        pthread_cond_init(&thread_data[i].pause_cond, NULL);
        pthread_mutex_init(&thread_data[i].parallel_value_mutex, NULL);
        pthread_create(&threads[i], NULL, ConcurrencyThreadFunc, &thread_data[i]);
    }
    int continueReceiving = 1;
    do {
        memset(buffer, 0, sizeof(buffer));
        int bytesReceived = recv(new_socket, buffer, sizeof(buffer), 0);
        if (bytesReceived <= 0) {
            printf("Error: Received no data from Python client or connection was closed.\n");
            continueReceiving = 0;
            break;
        }

        if(sscanf(buffer, "%d %d", &user_parallelism, &user_concurrency) != 2) {
            printf("Error: Failed to parse received data.\n");
            continueReceiving = 0;
            break;
        }
        if (atomic_load(&args.job_done)) {
            send(new_socket, termination_msg, sizeof(termination_msg), 0);
            break;
        }
        else{
            send(new_socket, ok_msg, sizeof(ok_msg), 0);
        }
        // Prompt user to enter values
        pthread_mutex_lock(&logMutex);
        fprintf(logFile, "Main Thread changed parallelism to %d and concurrency to %d\n", user_parallelism, user_concurrency);
        pthread_mutex_unlock(&logMutex);
        set_parallel_value(user_parallelism);
        set_concurrent_value(user_concurrency);
        sleep(2*UPDATE_TIME);

    } while(continueReceiving);

    printf("Main Thread change parallelism to MAX and concurrency to MAX\n");

    set_parallel_value(MAX_PARALLELISM);
    set_concurrent_value(MAX_CONCURRENCY);
    sleep(UPDATE_TIME);
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
    pthread_mutex_destroy(&parallelism_mutex);
    printf("All done\n");
    fclose(logFile);
    pthread_mutex_destroy(&logMutex);
    curl_global_cleanup();
    close(new_socket);
    close(server_fd);
    queue_destroy(files_need_to_be_downloaded);
    queue_destroy(files_downloaded);
    queue_destroy(generator_queue);
    queue_destroy(generator_queue_with_data_chunks);
    return 0;
}

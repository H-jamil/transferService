#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define CHUNK_SIZE 1048576 // 1 MB
#define FILE_PATH "/home/beams/MJAMIL/transferService/files_to_send/FILE0"
#define PORT 8080

typedef struct {
    int socket;
    long start;
    long end;
} ThreadData;

void* worker_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // Open file
    FILE* file = fopen(FILE_PATH, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return NULL;
    }

    // Seek to start of chunk
    if (fseek(file, data->start, SEEK_SET) != 0) {
        perror("Failed to seek file");
        fclose(file);
        return NULL;
    }

    // Read chunk
    size_t size = data->end - data->start;
    char* buffer = malloc(size);
    if (fread(buffer, 1, size, file) != size) {
        perror("Failed to read file");
        fclose(file);
        free(buffer);
        return NULL;
    }

    // Close file
    fclose(file);

    // Send chunk over socket
    if (send(data->socket, buffer, size, 0) == -1) {
        perror("Failed to send data");
    }

    // Clean up
    free(buffer);

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <IP> <num_threads>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *ip = argv[1];
    int num_threads = atoi(argv[2]);

    if (num_threads <= 0) {
        fprintf(stderr, "NUM_THREADS must be a positive integer\n");
        return 1;
    }

    // Get size of file
    FILE* file = fopen(FILE_PATH, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        return 1;
    }
    fseek(file, 0L, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

    // Create worker threads
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));

    // Create tasks
    ThreadData* tasks = malloc(num_threads * sizeof(ThreadData));
    for (int i = 0; i < num_threads; i++) {
        // Calculate start and end of chunk
        long start = (file_size / num_threads) * i;
        long end = (i == num_threads - 1) ? file_size : (file_size / num_threads) * (i + 1);

        // Create socket
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) {
            perror("Failed to create socket");
            free(tasks);
            free(threads);
            return 1;
        }

        // Setup server details
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(PORT);
        if (inet_pton(AF_INET, ip, &(server_addr.sin_addr)) <= 0) {
            perror("Invalid IP address");
            free(tasks);
            free(threads);
            close(sockfd);
            return 1;
        }

        // Connect to server
        if (connect(sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
            perror("Connection failed");
            free(tasks);
            free(threads);
            close(sockfd);
            return 1;
        }

        // Create task data
        tasks[i].socket = sockfd;
        tasks[i].start = start;
        tasks[i].end = end;

        // Start worker thread
        pthread_create(&threads[i], NULL, worker_thread, &tasks[i]);
    }

    // Wait for worker threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Clean up
    for (int i = 0; i < num_threads; i++) {
        close(tasks[i].socket);
    }
    printf("Sent file\n");
    int sockfd;
    struct sockaddr_in serv_addr;

    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

    // Setup server details
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, ip, &(serv_addr.sin_addr));

    // Connect to server
    connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr));

    // Send termination message
    char end_msg[CHUNK_SIZE] = "end";
    memset(end_msg + 3, 0, CHUNK_SIZE - 3); // fill the rest of the message with zeroes
    send(sockfd, end_msg, CHUNK_SIZE, 0);

    // Close the socket
    close(sockfd);
    printf("Sent termination message\n");

    free(tasks);
    free(threads);

    return 0;
}

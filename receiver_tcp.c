#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <pthread.h>

#define CHUNK_SIZE 1048576 // 1 MB
#define PORT 8080
#define FILE_PATH "./files_received/FILE0"

typedef struct {
    int sock;
} ThreadData;

void* worker_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    char buffer[CHUNK_SIZE];
    while (1) {
        ssize_t read_size = recv(data->sock, buffer, CHUNK_SIZE, 0);
        if (read_size < 0) {
            perror("Failed to receive data");
            return NULL;
        }
        else if (read_size == 0) {
            // Socket has been closed
            break;
        }

        if (strncmp(buffer, "end", 3) == 0) {
            // This is the termination message
            break;
        }
        else {
            // This is a data chunk, write it to the file
            FILE* file = fopen(FILE_PATH, "ab");
            if (file == NULL) {
                perror("Failed to open file");
                return NULL;
            }
            fwrite(buffer, 1, read_size, file);
            fclose(file);
        }
    }
    exit(EXIT_SUCCESS);
    // return 0;
}

int main(int argc, char* argv[]) {
    // Create socket
    int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_sock < 0) {
        perror("Failed to create socket");
        return 1;
    }

    // Set up server details
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(PORT);

    // Bind socket
    if (bind(listen_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Failed to bind socket");
        return 1;
    }

    // Listen for incoming connections
    if (listen(listen_sock, 3) < 0) {
        perror("Failed to listen");
        return 1;
    }

    while (1) {
        // Accept new connection
        struct sockaddr_in client_addr;
        socklen_t client_addr_len = sizeof(client_addr);
        int new_sock = accept(listen_sock, (struct sockaddr *)&client_addr, &client_addr_len);
        if (new_sock < 0) {
            perror("Failed to accept connection");
            return 1;
        }

        // Create new worker thread
        pthread_t thread_id;
        ThreadData* data = malloc(sizeof(ThreadData));
        data->sock = new_sock;
        if (pthread_create(&thread_id, NULL, worker_thread, data) < 0) {
            perror("Failed to create thread");
            return 1;
        }
    }

    return 0;
}

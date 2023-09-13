#include "/home/beams/MJAMIL/project/libzmq/include/zmq.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#define CHUNK_SIZE 1048576 // 1 MB
// #define OUTPUT_FILE "./files_received/FILE_R_0"
#define OUTPUT_FILE "output_file"
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <IP> <port>\n", argv[0]);
        exit(EXIT_FAILURE);
      }

    char *ip = argv[1];
    int port = atoi(argv[2]);

    char endpoint[128];
    sprintf(endpoint, "tcp://%s:%d", ip, port);

    // create ZeroMQ context
    void* context = zmq_ctx_new();

    // create ZeroMQ socket
    void* socket = zmq_socket(context, ZMQ_SUB);
    zmq_setsockopt(socket, ZMQ_SUBSCRIBE, "", 0);

    int rc = zmq_bind(socket, endpoint);
    assert(rc == 0);
    char buffer[CHUNK_SIZE];
    // open output file
    FILE* file = fopen(OUTPUT_FILE, "wb");
    if (file == NULL) {
        // fprintf(stderr, "Failed to open output file\n");
        perror("Failed to write to file");

        return 1;
    }

    while (1) {
        // receive chunk

        int size = zmq_recv(socket, buffer, CHUNK_SIZE, 0);
        if (size < 0) {
            perror("Failed to write to file");
            break;
        } else {
            // Print first 10 bytes of the buffer for debugging purposes.
            for (int i = 0; i < size && i < 10; ++i) {
                printf("buffer[%d] = %c\n", i, buffer[i]);
            }
        }

        // check for end message
        // check for end message
        if (size == CHUNK_SIZE && memcmp(buffer, "end", 3) == 0) {
            bool all_zeroes = true;
            for (int i = 3; i < CHUNK_SIZE; ++i) {
                if (buffer[i] != 0) {
                    all_zeroes = false;
                    break;
                }
            }
            if (all_zeroes) break;
        }

        // write chunk to file
        if (fwrite(buffer, 1, size, file) != size) {
            perror("Failed to write to file");
            break;
            }
    }

    // clean up
    fclose(file);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return 0;
}

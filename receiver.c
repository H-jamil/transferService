#include <stdio.h>
#include <zmq.h>
#include <assert.h>

#define CHUNK_SIZE 1048576 // 1 MB
#define OUTPUT_FILE "./files_received/FILE_R_0"

int main() {
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
    void* socket = zmq_socket(context, ZMQ_PULL);
    int rc = zmq_bind(socket, endpoint);
    assert(rc == 0);

    // open output file
    FILE* file = fopen(OUTPUT_FILE, "wb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }

    while (1) {
        // receive chunk
        char buffer[CHUNK_SIZE];
        int size = zmq_recv(socket, buffer, CHUNK_SIZE, 0);
        if (size < 0) {
            fprintf(stderr, "Failed to receive data\n");
            break;
        }

        // check for end message
        if (size == 3 && memcmp(buffer, "end", 3) == 0) {
          break;
        }

        // write chunk to file
        if (fwrite(buffer, 1, size, file) != size) {
            fprintf(stderr, "Failed to write to file\n");
            break;
        }
    }

    // clean up
    fclose(file);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return 0;
}

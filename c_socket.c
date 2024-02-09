#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h> // for memset
#include <unistd.h> // for close()
#include <stdio.h> // for printf
#include <stdlib.h> // for exit

#define PORT 8080


int main(int argc, char *argv[]) {
    int user_parallelism, user_concurrency;
    // Setup the socket to communicate with the Python program
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

    // ... [rest of the initializations] ...
// Replace the declaration of char continueInput = 'y'; with:
int continueReceiving = 1;

do {
    memset(buffer, 0, sizeof(buffer)); // Clear the buffer
    // Receive values from Python program
    int bytesReceived = recv(new_socket, buffer, sizeof(buffer), 0);

    // Check if the Python client has closed the connection
    if (bytesReceived <= 0) {
        continueReceiving = 0;
        break;
    }

    sscanf(buffer, "%d %d", &user_parallelism, &user_concurrency);
    printf("Received parallelism: %d, concurrency: %d\n", user_parallelism, user_concurrency);
    sleep(1);
    // ... [rest of your loop logic] ...

} while(continueReceiving);

    // Close the socket after the loop
    close(new_socket);
    close(server_fd);
    return 0;
    // ... [rest of your main function] ...
}

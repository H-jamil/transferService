#ifndef QUEUE_H
#define QUEUE_H

#include <stdatomic.h>

// Node structure for the queue
typedef struct Node {
    void* data;
    struct Node* next;
} Node;

// Queue structure
typedef struct Queue {
    Node* front;
    Node* rear;
    pthread_mutex_t lock;
    atomic_int size;
} Queue;

// Function prototypes for the queue operations

/**
 * Creates a new queue and returns a pointer to it.
 * Returns NULL on failure.
 */
Queue* queue_create();

/**
 * Destroys the given queue and frees all associated memory.
 */
void queue_destroy(Queue* queue);

/**
 * Pushes the given data into the queue.
 */
void queue_push(Queue* queue, void* data);

/**
 * Pops the front item from the queue and returns it.
 * Returns NULL if the queue is empty.
 */
void* queue_pop(Queue* queue);

/**
 * Returns the current size of the queue.
 */
int queue_size(Queue* queue);

#endif // QUEUE_H

#ifndef QUEUE_H
#define QUEUE_H

#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdatomic.h>

typedef struct Node {
    void* data;
    struct Node* next;
} Node;

typedef struct Queue {
    Node* front;
    Node* rear;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    atomic_int size;
} Queue;

Queue* queue_create();
void queue_destroy(Queue* queue);
void queue_push(Queue* queue, void* data);
void* queue_pop(Queue* queue);
int queue_size(Queue* queue);

#endif // QUEUE_H

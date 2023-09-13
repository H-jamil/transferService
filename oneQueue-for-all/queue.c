#include <stdlib.h>
#include <pthread.h>
#include "queue.h"

Queue* queue_create() {
    Queue* queue = malloc(sizeof(Queue));
    queue->front = NULL;
    queue->rear = NULL;
    pthread_mutex_init(&(queue->lock), NULL);
    pthread_cond_init(&(queue->cond), NULL);
    // queue->size = 0;
    atomic_init(&queue->size, 0); // Initialize size to 0
    return queue;
}

void queue_destroy(Queue* queue) {
    Node* current = queue->front;
    while (current != NULL) {
        Node* next = current->next;
        free(current);
        current = next;
    }
    pthread_mutex_destroy(&(queue->lock));
    pthread_cond_destroy(&(queue->cond));
    free(queue);
}

void queue_push(Queue* queue, void* data) {
    Node* new_node = malloc(sizeof(Node));
    new_node->data = data;
    new_node->next = NULL;

    pthread_mutex_lock(&(queue->lock));
    if (queue->rear != NULL) {
        queue->rear->next = new_node;
        queue->rear = new_node;
    } else {
        queue->front = new_node;
        queue->rear = new_node;
    }
    // queue->size++;
    atomic_fetch_add(&queue->size, 1); // Increment size atomically
    pthread_cond_signal(&(queue->cond));
    pthread_mutex_unlock(&(queue->lock));
}

void* queue_pop(Queue* queue) {
    pthread_mutex_lock(&(queue->lock));
    if (queue->front == NULL) {
        // pthread_cond_wait(&(queue->cond), &(queue->lock));
        pthread_mutex_unlock(&(queue->lock));
        return NULL;
    }
    Node* front_node = queue->front;
    void* data = front_node->data;
    queue->front = front_node->next;
    if (queue->front == NULL) {
        queue->rear = NULL;
    }
    atomic_fetch_sub(&queue->size, 1); // Decrement size atomically
    pthread_mutex_unlock(&(queue->lock));
    free(front_node);
    return data;
}

int queue_size(Queue* queue) {
    // pthread_mutex_lock(&(queue->lock));
    // int size = queue->size;
    // pthread_mutex_unlock(&(queue->lock));
    // return size;
    return atomic_load(&queue->size); // Load the current size atomically

}

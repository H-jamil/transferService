#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
int call_value_hash=0;

// Define the structure for list H_Node in the chain
typedef struct H_Node {
    void *item;
    struct H_Node *next;
} H_Node;

// Define the structure for hash table
typedef struct HashTable {
    H_Node **buckets;
    size_t size;
} HashTable;

// Function to create a new hash table
HashTable* hash_table_create(size_t size) {
    HashTable *newTable = malloc(sizeof(HashTable));
    newTable->buckets = malloc(sizeof(H_Node*) * size);
    memset(newTable->buckets, 0, sizeof(H_Node*) * size);
    newTable->size = size;
    return newTable;
}

// Simple hash function: uses the address of the object to create a hash
unsigned int hash(void *item, size_t size) {
    return ((unsigned long)item) % size;
}

// Function to add an item to the hash table
void hash_table_add(HashTable *table, void *item) {
    unsigned int bucket = hash(item, table->size);
    // unsigned int bucket =call_value_hash;
    H_Node *newNode = malloc(sizeof(H_Node));
    newNode->item = item;
    newNode->next = table->buckets[bucket];
    table->buckets[bucket] = newNode;
    call_value_hash++;
}

// Function to check if an item exists in the hash table
bool hash_table_contains(HashTable *table, void *item) {
    unsigned int bucket = hash(item, table->size);
    H_Node *current = table->buckets[bucket];
    while (current) {
        if (current->item == item) {
            return true;
        }
        current = current->next;
    }
    return false;
}

// Function to remove an item from the hash table
void hash_table_remove(HashTable *table, void *item) {
    unsigned int bucket = hash(item, table->size);
    H_Node *current = table->buckets[bucket];
    H_Node *prev = NULL;
    while (current) {
        if (current->item == item) {
            if (prev) {
                prev->next = current->next;
            } else {
                table->buckets[bucket] = current->next;
            }
            free(current);
            return;
        }
        prev = current;
        current = current->next;
    }
}

// Function to destroy the hash table
void hash_table_destroy(HashTable *table) {
    for (size_t i = 0; i < table->size; ++i) {
        H_Node *current = table->buckets[i];
        while (current) {
            H_Node *temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(table->buckets);
    free(table);
}


// Function to print pointers stored in the hash table
void hash_table_print_pointers(HashTable *table) {
    if (!table) {
        printf("Hash table is NULL.\n");
        return;
    }

    printf("Hash Table Pointers:\n");
    for (size_t i = 0; i < table->size; ++i) {
        H_Node *node = table->buckets[i];
        if (node) {
            printf("Bucket %lu: ", i);
            while (node) {
                printf("%p ", node->item);
                node = node->next;
            }
            printf("\n");
        } else {
            printf("Bucket %lu: EMPTY\n", i);
        }
    }
}

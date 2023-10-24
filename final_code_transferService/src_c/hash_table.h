#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <stdlib.h>
#include <stdbool.h>

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
HashTable* hash_table_create(size_t size);

// Function to add an item to the hash table
void hash_table_add(HashTable *table, void *item);

// Function to check if an item exists in the hash table
bool hash_table_contains(HashTable *table, void *item);

// Function to remove an item from the hash table
void hash_table_remove(HashTable *table, void *item);

// Function to destroy the hash table
void hash_table_destroy(HashTable *table);

void hash_table_print_pointers(HashTable *table);


#endif // HASH_TABLE_H

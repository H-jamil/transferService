#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include <stdlib.h>
#include <pthread.h>
#include <pthread.h>
#include <stdint.h>
#include <sched.h>
#include <assert.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/time.h>

typedef struct parallel_work_data{

  char *url;
  long start_byte;
  long end_byte;
  char *output_file;

}parallel_work_data;


typedef struct DataGenerator{

  char *url;
  long start_byte;
  long end_byte;
  char *output_file;
  size_t dataSize;
  size_t chunk_size;
  size_t currentIndex;
  int current_call_number;
  int finished;
  pthread_mutex_t mutex_generator;
  pthread_mutex_t mutex_generator_finished;
}DataGenerator;

bool is_finished(DataGenerator *gen);
DataGenerator* data_generator_init(char *url, char *o_file, size_t dataSize, size_t chunkSize);
parallel_work_data* data_generator_next(DataGenerator *gen);
void data_generator_free(DataGenerator *gen);


#endif // DATA_GENERATOR_H

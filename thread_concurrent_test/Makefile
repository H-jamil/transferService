# Compiler
CC = gcc

# Compiler Flags
CFLAGS = -Wall

# Sources
SOURCES = main.c concurrent_thread.c parallel_thread.c data_generator.c queue.c

# Object Files
OBJ = $(SOURCES:.c=.o)

# Executable
EXECUTABLE = parallel_concurrent

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lpthread -lcurl
	rm -f $(OBJ)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXECUTABLE)

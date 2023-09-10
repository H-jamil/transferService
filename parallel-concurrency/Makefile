# Compiler
CC = gcc

# Compiler Flags
CFLAGS = -Wall -g

# Sources
SOURCES = main.c parallel.c concurrency.c data_generator.c

# Object Files
OBJ = $(SOURCES:.c=.o)

# Executable
EXECUTABLE = program

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lpthread -lcurl
	rm -f $(OBJ)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXECUTABLE)

/*
 * mm-naive.c - The fastest, least memory-efficient malloc package.
 *
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * NOTE TO STUDENTS: Replace this header comment with your own header
 * comment that gives a high level description of your solution.
 */

// EVOLVE-BLOCK-START
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "ateam",
    /* First member's full name */
    "Harry Bovik",
    /* First member's email address */
    "bovik@cs.cmu.edu",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""};

/* 16 bytes alignment */
#define ALIGNMENT 16

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))

/* Minimal block header to track allocated blocks */
typedef struct {
    size_t size;
    int free;
} block_header_t;

/* Free list node for tracking free blocks */
typedef struct free_node {
    struct free_node *next;
    void *ptr;
} free_node_t;

/* 
 * mm_init - initialize the malloc package.
 * Sets up a pointer to track the start of allocated memory and free list.
 */
static void *heap_start = NULL;
static free_node_t *free_list = NULL;

int mm_init(void) { 
    heap_start = NULL;
    free_list = NULL;
    return 0; 
}

/*
 * mm_malloc - Allocate a block by first checking free list, then brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size) {
    if (size == 0)
        return NULL;
    
    /* Add space for block header */
    size_t total_size = ALIGN(sizeof(block_header_t)) + ALIGN(size);
    
    /* First try to find a free block that fits */
    free_node_t **prev = &free_list;
    free_node_t *current = free_list;
    
    while (current) {
        /* Check if this free block is big enough */
        char *block_start = (char *)current->ptr - ALIGN(sizeof(block_header_t));
        block_header_t *header = (block_header_t *)block_start;
        if (header->size >= size) {
            /* Found a suitable block, remove from free list */
            *prev = current->next;
            free(current);
            
            /* Mark as allocated */
            header->free = 0;
            /* Return the payload area which should be properly aligned */
            return (void *)((char *)block_start + ALIGN(sizeof(block_header_t)));
        }
        prev = &current->next;
        current = current->next;
    }
    
    /* No suitable free block found, allocate new memory */
    void *p = mem_sbrk(total_size);
    
    if (p == (void *)-1)
        return NULL;
    
    /* Initialize block header */
    block_header_t *header = (block_header_t *)p;
    header->size = size;
    header->free = 0;
    
    /* Return the payload area, which should be properly aligned */
    return (void *)((char *)p + ALIGN(sizeof(block_header_t)));
}

/*
 * mm_free - Mark block as free and add to free list
 */
void mm_free(void *ptr) {
    if (ptr == NULL)
        return;
    
    /* Get block header */
    block_header_t *header = (block_header_t *)((char *)ptr - ALIGN(sizeof(block_header_t)));
    
    /* Only add to free list if it's a reasonably sized block */
    if (header->size >= 16) {
        header->free = 1;
        
        /* Add to free list */
        free_node_t *node = (free_node_t *)malloc(sizeof(free_node_t));
        if (node) {
            node->ptr = ptr;
            node->next = free_list;
            free_list = node;
        }
    }
}

/*
 * mm_realloc - Implemented with free list support
 */
void *mm_realloc(void *ptr, size_t size) {
    if (ptr == NULL)
        return mm_malloc(size);
    
    if (size == 0) {
        mm_free(ptr);
        return NULL;
    }
    
    void *newptr = mm_malloc(size);
    if (newptr == NULL)
        return NULL;
    
    /* Get original block size from header */
    block_header_t *header = (block_header_t *)((char *)ptr - ALIGN(sizeof(block_header_t)));
    size_t copySize = header->size;
    
    if (size < copySize)
        copySize = size;
    
    memcpy(newptr, ptr, copySize);
    mm_free(ptr);
    return newptr;
}

// EVOLVE-BLOCK-END

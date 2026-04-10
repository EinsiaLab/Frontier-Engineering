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

/* Basic block header structure */
typedef struct header {
    size_t size;
    struct header *next;
} header_t;

/* Free list head */
static header_t *free_list = NULL;

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) {
    free_list = NULL;
    return 0;
}

/*
 * mm_malloc - Allocate a block by using free list if possible, otherwise increment brk.
 */
void *mm_malloc(size_t size) {
    if (size == 0) return NULL;
    
    size_t total_size = ALIGN(size + SIZE_T_SIZE);
    header_t *prev = NULL;
    header_t *curr = free_list;
    header_t *best = NULL;
    header_t *best_prev = NULL;
    size_t best_diff = 0;

    /* Search free list for best-fit (smallest block that fits) */
    while (curr != NULL) {
        if (curr->size >= total_size) {
            size_t diff = curr->size - total_size;
            if (best == NULL || diff < best_diff) {
                best = curr;
                best_prev = prev;
                best_diff = diff;
            }
        }
        prev = curr;
        curr = curr->next;
    }
    
    if (best != NULL) {
        /* Found a suitable block */
        if (best->size >= total_size + SIZE_T_SIZE + ALIGNMENT) {
            /* Split the block if remaining space is enough for a new block with header */
            header_t *new_block = (header_t *)((char *)best + total_size);
            size_t new2_size = best->size - total_size;
            new_block->size = new2_size;
            new_block->next = best->next;
            best->size = total_size;
            if (best_prev) {
                best_prev->next = new_block;
            } else {
                free_list = new_block;
            }
        } else {
            /* Remove the block from free list */
            if (best_prev) {
                best_prev->next = best->next;
            } else {
                free_list = best->next;
            }
        }
        /* Mark as allocated by storing size at start */
        *(size_t *)best = size;
        return (void *)((char *)best + SIZE_T_SIZE);
    }

    /* No suitable free block found, allocate new memory */
    void *p = mem_sbrk(total_size);
    if (p == (void *)-1)
        return NULL;
    else {
        header_t *block = (header_t *)p;
        block->size = total_size;
        *(size_t *)block = size;
        return (void *)((char *)block + SIZE_T_SIZE);
    }
}

/*
 * mm_free - Add block to free list and coalesce with adjacent free blocks.
 */
void mm_free(void *ptr) {
    if (ptr == NULL) return;
    
    header_t *block = (header_t *)((char *)ptr - SIZE_T_SIZE);
    /* Recover total size from stored user size */
    size_t user_size = *(size_t *)block;
    block->size = ALIGN(user_size + SIZE_T_SIZE);
    
    /* Coalesce with adjacent free blocks in the list */
    header_t *curr = free_list;
    header_t *prev = NULL;
    
    /* Sort by address to enable coalescing */
    while (curr != NULL && (void *)curr < (void *)block) {
        prev = curr;
        curr = curr->next;
    }
    
    /* Check if previous block is adjacent */
    if (prev != NULL && (char *)prev + prev->size == (char *)block) {
        /* Merge with previous block */
        prev->size += block->size;
        block = prev;
    } else {
        /* Insert block into free list */
        block->next = curr;
        if (prev) {
            prev->next = block;
        } else {
            free_list = block;
        }
    }
    
    /* Check if next block is adjacent */
    if (curr != NULL && (char *)block + block->size == (char *)curr) {
        /* Merge with next block */
        block->size += curr->size;
        block->next = curr->next;
    }
}

/*
 * mm_realloc - Reallocate memory block with in-place expansion when possible
 */
void *mm_realloc(void *ptr, size_t size) {
    if (ptr == NULL) {
        return mm_malloc(size);
    }
    if (size == 0) {
        mm_free(ptr);
        return NULL;
    }
    
    header_t *block = (header_t *)((char *)ptr - SIZE_T_SIZE);
    size_t old_size = *(size_t *)block;
    size_t total_size = ALIGN(size + SIZE_T_SIZE);
    size_t old_total_size = ALIGN(old_size + SIZE_T_SIZE);
    
    /* Case 1: Same size or smaller (within same block size due to alignment) */
    if (total_size <= old_total_size) {
        /* Just update the stored size */
        *(size_t *)block = size;
        return ptr;
    }
    
    /* Case 2: Check if next block is free and adjacent, and combined size is sufficient */
    header_t *curr = free_list;
    header_t *prev = NULL;
    while (curr != NULL) {
        if ((char *)block + old_total_size == (char *)curr) {
            /* Next block is adjacent and free */
            size_t combined_size = old_total_size + curr->size;
            if (combined_size >= total_size) {
                /* Can expand in place */
                /* Remove curr from free list */
                if (prev) {
                    prev->next = curr->next;
                } else {
                    free_list = curr->next;
                }
                /* Update block size */
                block->size = combined_size;
                *(size_t *)block = size;
                /* If there's leftover space, split */
                if (combined_size >= total_size + SIZE_T_SIZE + ALIGNMENT) {
                    header_t *new_free = (header_t *)((char *)block + total_size);
                    new_free->size = combined_size - total_size;
                    /* Insert new_free into free list */
                    new_free->next = free_list;
                    free_list = new_free;
                }
                return ptr;
            }
        }
        prev = curr;
        curr = curr->next;
    }
    
    /* Case 3: Otherwise, allocate new block and copy */
    void *newptr = mm_malloc(size);
    if (newptr == NULL)
        return NULL;
    size_t copySize = old_size;
    if (size < copySize)
        copySize = size;
    memcpy(newptr, ptr, copySize);
    mm_free(ptr);
    return newptr;
}

// EVOLVE-BLOCK-END

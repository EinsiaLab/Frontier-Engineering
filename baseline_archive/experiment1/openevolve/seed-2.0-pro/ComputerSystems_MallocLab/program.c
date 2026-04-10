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

/* Free list block structure (single-linked for simplicity) */
typedef struct block {
    size_t size;
    struct block *next;
} block_t;

#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))
static block_t *free_list = NULL;

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) { 
    free_list = NULL;
    return 0; 
}

/*
 * mm_malloc - Allocate a block by first reusing freed blocks if possible,
 *     otherwise increment the brk pointer.
 */
void *mm_malloc(size_t size) {
  int newsize = ALIGN(size + SIZE_T_SIZE);
  block_t *curr = free_list;
  block_t *prev = NULL;

  /* First-fit search for suitable free block */
  while (curr) {
    if (curr->size >= size) {
      /* Remove block from free list and return to user */
      if (prev) prev->next = curr->next;
      else free_list = curr->next;
      return (void *)((char *)curr + SIZE_T_SIZE);
    }
    prev = curr;
    curr = curr->next;
  }

  /* No free block available, extend heap */
  void *p = mem_sbrk(newsize);
  if (p == (void *)-1)
    return NULL;
  else {
    *(size_t *)p = size;
    return (void *)((char *)p + SIZE_T_SIZE);
  }
}

/*
 * mm_free - Add freed block to free list for future reuse
 */
void mm_free(void *ptr) {
    if (!ptr) return;
    block_t *block = (block_t*)((char*)ptr - SIZE_T_SIZE);
    block->next = free_list;
    free_list = block;
}

/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
void *mm_realloc(void *ptr, size_t size) {
  void *oldptr = ptr;
  void *newptr;
  size_t copySize;

  newptr = mm_malloc(size);
  if (newptr == NULL)
    return NULL;
  copySize = *(size_t *)((char *)oldptr - SIZE_T_SIZE);
  if (size < copySize)
    copySize = size;
  memcpy(newptr, oldptr, copySize);
  mm_free(oldptr);
  return newptr;
}

// EVOLVE-BLOCK-END

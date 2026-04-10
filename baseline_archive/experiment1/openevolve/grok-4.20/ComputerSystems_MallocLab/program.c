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
#include <stdlib.h>
#include <string.h>

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

struct block {
    size_t size;
    struct block *next;
};

static struct block *free_list = NULL;

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) {
    free_list = NULL;
    return 0;
}

/*
 * mm_malloc - Allocate a block by incrementing the brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size) {
  if (size == 0)
    size = 1;  /* malloc(0) should return a valid (non-NULL) pointer */

  int newsize = ALIGN(size + SIZE_T_SIZE);

  /* First-fit search of free list */
  struct block **prev = &free_list;
  for (struct block *p = free_list; p != NULL; p = p->next) {
    if (p->size >= newsize) {
      *prev = p->next;
      return (void *)((char *)p + SIZE_T_SIZE);
    }
    prev = &(p->next);
  }

  /* No fit, extend the heap */
  void *p = mem_sbrk(newsize);
  if (p == (void *)-1)
    return NULL;
  else {
    *(size_t *)p = newsize;
    return (void *)((char *)p + SIZE_T_SIZE);
  }
}

/*
 * mm_free - Freeing a block adds it to the free list.
 */
void mm_free(void *ptr) {
  if (ptr == NULL)
    return;

  struct block *bp = (struct block *)((char *)ptr - SIZE_T_SIZE);
  bp->size = *(size_t *)bp;
  bp->next = free_list;
  free_list = bp;
}

/*
 * mm_realloc - Improved version that reuses the block when possible,
 * extends the block in-place when it is at the end of the heap (avoids
 * temporary extra memory usage that was causing OOM), and falls back
 * to alloc/copy/free only when necessary. This should fix realloc
 * trace failures.
 */
void *mm_realloc(void *ptr, size_t size) {
  if (ptr == NULL) {
    return mm_malloc(size);
  }
  if (size == 0) {
    mm_free(ptr);
    return NULL;
  }

  void *header = (char *)ptr - SIZE_T_SIZE;
  size_t oldsize = *(size_t *)header;
  size_t old_payload = oldsize - SIZE_T_SIZE;

  if (size <= old_payload) {
    return ptr;
  }

  /* Try to extend the block in place if it is the last block in the heap */
  if ((char *)header + oldsize == (char *)mem_sbrk(0)) {
    size_t newsize = ALIGN(size + SIZE_T_SIZE);
    size_t incr = newsize - oldsize;
    if (incr > 0) {
      if (mem_sbrk((int)incr) == (void *)-1) {
        return NULL;
      }
      *(size_t *)header = newsize;
      return ptr;
    }
  }

  /* Fallback: allocate new block, copy, free old */
  void *newptr = mm_malloc(size);
  if (newptr == NULL)
    return NULL;
  size_t copySize = old_payload;
  if (size < copySize)
    copySize = size;
  memcpy(newptr, ptr, copySize);
  mm_free(ptr);
  return newptr;
}

// EVOLVE-BLOCK-END

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

#define ALIGNMENT 16
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Word and double word sizes */
#define WSIZE 4
#define DSIZE 8
#define OVERHEAD 16  /* header + footer = 8 bytes, but min block for alignment */
#define CHUNKSIZE (1 << 12)

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Pack size and allocated bit into a word */
#define PACK(size, alloc) ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p) (*(unsigned int *)(p))
#define PUT(p, val) (*(unsigned int *)(p) = (val))

/* Read size and allocated fields from address p */
#define GET_SIZE(p) (GET(p) & ~0xF)
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp) ((char *)(bp) - WSIZE)
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))

/* Free list pointers stored in payload area */
#define NEXT_FREE(bp) (*(void **)(bp))
#define PREV_FREE(bp) (*(void **)((char *)(bp) + sizeof(void *)))

#define NUM_CLASSES 10

static char *heap_listp = NULL;
static void *free_lists[NUM_CLASSES];

/* Get size class index for a given size */
static int get_class(size_t size) {
  if (size <= 32) return 0;
  if (size <= 64) return 1;
  if (size <= 128) return 2;
  if (size <= 256) return 3;
  if (size <= 512) return 4;
  if (size <= 1024) return 5;
  if (size <= 2048) return 6;
  if (size <= 4096) return 7;
  if (size <= 8192) return 8;
  return 9;
}

static void insert_free(void *bp) {
  int cls = get_class(GET_SIZE(HDRP(bp)));
  NEXT_FREE(bp) = free_lists[cls];
  PREV_FREE(bp) = NULL;
  if (free_lists[cls] != NULL)
    PREV_FREE(free_lists[cls]) = bp;
  free_lists[cls] = bp;
}

static void remove_free(void *bp) {
  int cls = get_class(GET_SIZE(HDRP(bp)));
  void *prev = PREV_FREE(bp);
  void *next = NEXT_FREE(bp);
  if (prev)
    NEXT_FREE(prev) = next;
  else
    free_lists[cls] = next;
  if (next)
    PREV_FREE(next) = prev;
}

static void *coalesce(void *bp) {
  size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
  size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
  size_t size = GET_SIZE(HDRP(bp));

  if (prev_alloc && next_alloc) {
    /* nothing */
  } else if (prev_alloc && !next_alloc) {
    remove_free(NEXT_BLKP(bp));
    size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
  } else if (!prev_alloc && next_alloc) {
    remove_free(PREV_BLKP(bp));
    size += GET_SIZE(HDRP(PREV_BLKP(bp)));
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
    bp = PREV_BLKP(bp);
  } else {
    remove_free(PREV_BLKP(bp));
    remove_free(NEXT_BLKP(bp));
    size += GET_SIZE(HDRP(PREV_BLKP(bp))) + GET_SIZE(FTRP(NEXT_BLKP(bp)));
    PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
    PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
    bp = PREV_BLKP(bp);
  }
  insert_free(bp);
  return bp;
}

static void *extend_heap(size_t words) {
  char *bp;
  size_t size = (words % 2) ? (words + 1) * WSIZE : words * WSIZE;
  size = ALIGN(size);
  if (size < ALIGNMENT * 2)
    size = ALIGNMENT * 2;
  if ((long)(bp = mem_sbrk(size)) == -1)
    return NULL;
  PUT(HDRP(bp), PACK(size, 0));
  PUT(FTRP(bp), PACK(size, 0));
  PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* new epilogue */
  return coalesce(bp);
}

int mm_init(void) {
  /* Allocate enough initial space so that the first block payload is 16-byte aligned.
   * We need: padding words + prologue header(4) + prologue footer(4) + epilogue header(4).
   * After mem_sbrk, heap starts at some address. We add 6 words (24 bytes) of padding
   * so that the first regular block's payload (after a 4-byte header) is 16-byte aligned.
   */
  int i;
  for (i = 0; i < NUM_CLASSES; i++)
    free_lists[i] = NULL;

  if ((heap_listp = mem_sbrk(8 * WSIZE)) == (void *)-1)
    return -1;
  PUT(heap_listp + (0 * WSIZE), 0);                    /* padding */
  PUT(heap_listp + (1 * WSIZE), 0);                    /* padding */
  PUT(heap_listp + (2 * WSIZE), 0);                    /* padding */
  PUT(heap_listp + (3 * WSIZE), PACK(ALIGNMENT, 1));   /* prologue header */
  PUT(heap_listp + (4 * WSIZE), 0);                    /* prologue payload */
  PUT(heap_listp + (5 * WSIZE), 0);                    /* prologue payload */
  PUT(heap_listp + (6 * WSIZE), PACK(ALIGNMENT, 1));   /* prologue footer */
  PUT(heap_listp + (7 * WSIZE), PACK(0, 1));           /* epilogue header */
  heap_listp += (4 * WSIZE);

  if (extend_heap(CHUNKSIZE / WSIZE) == NULL)
    return -1;
  return 0;
}

static void *find_fit(size_t asize) {
  int cls = get_class(asize);
  void *bp;
  void *best = NULL;
  size_t best_size = (size_t)-1;
  int i;
  for (i = cls; i < NUM_CLASSES; i++) {
    int count = 0;
    for (bp = free_lists[i]; bp != NULL && count < 30; bp = NEXT_FREE(bp), count++) {
      size_t bsize = GET_SIZE(HDRP(bp));
      if (bsize >= asize) {
        if (bsize == asize) return bp;
        if (bsize < best_size) {
          best = bp;
          best_size = bsize;
        }
      }
    }
    if (best != NULL) return best;
  }
  return best;
}

static void *place(void *bp, size_t asize) {
  size_t csize = GET_SIZE(HDRP(bp));
  size_t remainder = csize - asize;
  remove_free(bp);

  if (remainder >= (ALIGNMENT * 2)) {
    if (asize <= 96 && remainder >= 96) {
      /* Small alloc: place at END, keep free block at front */
      PUT(HDRP(bp), PACK(remainder, 0));
      PUT(FTRP(bp), PACK(remainder, 0));
      insert_free(bp);
      bp = NEXT_BLKP(bp);
      PUT(HDRP(bp), PACK(asize, 1));
      PUT(FTRP(bp), PACK(asize, 1));
    } else {
      PUT(HDRP(bp), PACK(asize, 1));
      PUT(FTRP(bp), PACK(asize, 1));
      void *nbp = NEXT_BLKP(bp);
      PUT(HDRP(nbp), PACK(remainder, 0));
      PUT(FTRP(nbp), PACK(remainder, 0));
      insert_free(nbp);
    }
  } else {
    PUT(HDRP(bp), PACK(csize, 1));
    PUT(FTRP(bp), PACK(csize, 1));
  }
  return bp;
}

void *mm_malloc(size_t size) {
  size_t asize;
  size_t extendsize;
  char *bp;

  if (size == 0)
    return NULL;

  asize = ALIGN(size + DSIZE);
  if (asize < ALIGNMENT * 2)
    asize = ALIGNMENT * 2;

  if ((bp = find_fit(asize)) != NULL) {
    bp = place(bp, asize);
    return bp;
  }

  /* Use smaller extension for large requests to reduce waste */
  if (asize >= CHUNKSIZE)
    extendsize = asize;
  else
    extendsize = MAX(asize, CHUNKSIZE);
  if ((bp = extend_heap(extendsize / WSIZE)) == NULL)
    return NULL;
  bp = place(bp, asize);
  return bp;
}

void mm_free(void *ptr) {
  if (ptr == NULL)
    return;
  size_t size = GET_SIZE(HDRP(ptr));
  PUT(HDRP(ptr), PACK(size, 0));
  PUT(FTRP(ptr), PACK(size, 0));
  coalesce(ptr);
}

void *mm_realloc(void *ptr, size_t size) {
  if (ptr == NULL)
    return mm_malloc(size);
  if (size == 0) {
    mm_free(ptr);
    return NULL;
  }

  size_t asize = ALIGN(size + DSIZE);
  if (asize < ALIGNMENT * 2)
    asize = ALIGNMENT * 2;
  size_t oldsize = GET_SIZE(HDRP(ptr));

  if (oldsize >= asize) {
    /* Could split, but just return for now */
    return ptr;
  }

  /* Check if next block is free and combined size is enough */
  void *next = NEXT_BLKP(ptr);
  size_t next_size = GET_SIZE(HDRP(next));
  if (!GET_ALLOC(HDRP(next)) && (oldsize + next_size) >= asize) {
    remove_free(next);
    PUT(HDRP(ptr), PACK(oldsize + next_size, 1));
    PUT(FTRP(ptr), PACK(oldsize + next_size, 1));
    return ptr;
  }

  /* Check if next block is free and AFTER that is epilogue - combine both */
  if (!GET_ALLOC(HDRP(next)) && GET_SIZE(HDRP(NEXT_BLKP(next))) == 0) {
    size_t combined = oldsize + next_size;
    size_t extend = (asize > combined) ? (asize - combined) : 0;
    extend = ALIGN(extend);
    if (extend < ALIGNMENT * 2)
      extend = ALIGNMENT * 2;
    remove_free(next);
    void *p = mem_sbrk(extend);
    if ((long)p == -1)
      return NULL;
    PUT(HDRP(ptr), PACK(combined + extend, 1));
    PUT(FTRP(ptr), PACK(combined + extend, 1));
    PUT(HDRP(NEXT_BLKP(ptr)), PACK(0, 1));
    return ptr;
  }

  /* Check if next block is epilogue (end of heap) - extend */
  if (next_size == 0) {
    size_t extend = asize - oldsize;
    extend = ALIGN(extend);
    if (extend < ALIGNMENT * 2)
      extend = ALIGNMENT * 2;
    void *p = mem_sbrk(extend);
    if ((long)p == -1)
      return NULL;
    PUT(HDRP(ptr), PACK(oldsize + extend, 1));
    PUT(FTRP(ptr), PACK(oldsize + extend, 1));
    PUT(HDRP(NEXT_BLKP(ptr)), PACK(0, 1));
    return ptr;
  }

  void *newptr = mm_malloc(size);
  if (newptr == NULL)
    return NULL;
  memcpy(newptr, ptr, oldsize - DSIZE);
  mm_free(ptr);
  return newptr;
}

// EVOLVE-BLOCK-END

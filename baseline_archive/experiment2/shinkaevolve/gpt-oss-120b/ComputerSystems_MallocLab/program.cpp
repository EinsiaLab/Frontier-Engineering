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

/* Basic constants and macros */
#define WSIZE       8       /* Word and header/footer size (bytes) */
#define DSIZE       16      /* Double word size (bytes) */
#define CHUNKSIZE   (1<<12) /* Extend heap by this amount (bytes) */
#define MIN_BLOCK   32      /* Minimum block size: hdr + ftr + 2 ptrs */
#define SPLIT_THRESH 48     /* Minimum remainder size to split block */

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Pack a size and allocated bit into a word */
#define PACK(size, alloc)  ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p)       (*(unsigned int *)(p))
#define PUT(p, val)  (*(unsigned int *)(p) = (val))

/* Read the size and allocated fields from address p */
#define GET_SIZE(p)  (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp)       ((char *)(bp) - WSIZE)
#define FTRP(bp)       ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp)  ((char *)(bp) + GET_SIZE(((char *)(bp) - WSIZE)))
#define PREV_BLKP(bp)  ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))

/* Explicit free list pointer macros - stored in payload of free blocks */
#define PRED_PTR(bp)  ((char *)(bp))
#define SUCC_PTR(bp)  ((char *)(bp) + WSIZE)
#define PRED(bp)      (*(char **)(PRED_PTR(bp)))
#define SUCC(bp)      (*(char **)(SUCC_PTR(bp)))

/* Quick-fit sizes: 32, 48, 64, 80, 96, 112, 128 bytes */
#define NUM_QUICK_LISTS  7
#define QUICK_MAX_SIZE   128

/* Segregated lists for larger blocks */
#define NUM_SEG_LISTS    10
#define NUM_LISTS        (NUM_QUICK_LISTS + NUM_SEG_LISTS)

/* Quick-fit list indices */
#define QUICK_INDEX(size) (((size) - MIN_BLOCK) / 16)

/* Global variables */
static char *heap_listp;
static char *free_lists[NUM_LISTS];  /* Quick-fit + Segregated free lists */

/* Function prototypes */
static void *extend_heap(size_t words);
static void place(void *bp, size_t asize);
static void *find_fit(size_t asize);
static void *coalesce(void *bp);
static void insert_to_list(void *bp, int index);
static void remove_from_list(void *bp, int index);
static int get_list_index(size_t size);

/*
 * insert_to_list - Insert block at front of specified free list (LIFO)
 */
static void insert_to_list(void *bp, int index) {
    char *list = free_lists[index];

    PRED(bp) = NULL;
    SUCC(bp) = list;

    if (list != NULL) {
        PRED(list) = bp;
    }
    free_lists[index] = bp;
}

/*
 * remove_from_list - Remove block from specified free list
 */
static void remove_from_list(void *bp, int index) {
    char *pred = PRED(bp);
    char *succ = SUCC(bp);

    if (pred == NULL) {
        free_lists[index] = succ;
    } else {
        SUCC(pred) = succ;
    }

    if (succ != NULL) {
        PRED(succ) = pred;
    }
}

/*
 * get_list_index - Get the free list index for a given size
 * Returns 0-6 for quick-fit (32-128 bytes)
 * Returns 7-16 for segregated fits (larger blocks)
 */
static int get_list_index(size_t size) {
    if (size <= QUICK_MAX_SIZE) {
        /* Quick-fit: exact size classes for 32, 48, 64, 80, 96, 112, 128 */
        int idx = QUICK_INDEX(size);
        if (idx < NUM_QUICK_LISTS) {
            return idx;
        }
    }

    /* Segregated fit for larger blocks */
    int index = NUM_QUICK_LISTS;
    size = size >> 7;  /* Divide by 128 */
    while (size > 1 && index < NUM_LISTS - 1) {
        size >>= 1;
        index++;
    }
    return index;
}

/*
 * mm_init - Initialize the memory manager.
 */
int mm_init(void) {
    int i;

    /* Initialize all free lists */
    for (i = 0; i < NUM_LISTS; i++) {
        free_lists[i] = NULL;
    }

    /* Create the initial empty heap */
    if ((heap_listp = mem_sbrk(4 * WSIZE)) == (void *)-1)
        return -1;
    PUT(heap_listp, 0);                            /* Alignment padding */
    PUT(heap_listp + (1 * WSIZE), PACK(DSIZE, 1)); /* Prologue header */
    PUT(heap_listp + (2 * WSIZE), PACK(DSIZE, 1)); /* Prologue footer */
    PUT(heap_listp + (3 * WSIZE), PACK(0, 1));     /* Epilogue header */
    heap_listp += (2 * WSIZE);

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE / WSIZE) == NULL)
        return -1;
    return 0;
}

/*
 * extend_heap - Extend heap with free block and return its block pointer.
 */
static void *extend_heap(size_t words) {
    char *bp;
    size_t size;

    /* Allocate an even number of words to maintain alignment */
    size = (words % 2) ? (words + 1) * WSIZE : words * WSIZE;
    if ((long)(bp = mem_sbrk(size)) == -1)
        return NULL;

    /* Initialize free block header/footer and the epilogue header */
    PUT(HDRP(bp), PACK(size, 0));         /* Free block header */
    PUT(FTRP(bp), PACK(size, 0));         /* Free block footer */
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* New epilogue header */

    /* Coalesce if the previous block was free */
    return coalesce(bp);
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block.
 */
static void *coalesce(void *bp) {
    size_t prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)));
    size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    size_t size = GET_SIZE(HDRP(bp));

    if (prev_alloc && next_alloc) {            /* Case 1 */
        int index = get_list_index(size);
        insert_to_list(bp, index);
        return bp;
    } else if (prev_alloc && !next_alloc) {    /* Case 2 */
        size_t next_size = GET_SIZE(HDRP(NEXT_BLKP(bp)));
        int next_idx = get_list_index(next_size);
        size += next_size;
        remove_from_list(NEXT_BLKP(bp), next_idx);
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));
        int index = get_list_index(size);
        insert_to_list(bp, index);
    } else if (!prev_alloc && next_alloc) {    /* Case 3 */
        size_t prev_size = GET_SIZE(HDRP(PREV_BLKP(bp)));
        int prev_idx = get_list_index(prev_size);
        size += prev_size;
        remove_from_list(PREV_BLKP(bp), prev_idx);
        PUT(FTRP(bp), PACK(size, 0));
        PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
        bp = PREV_BLKP(bp);
        int index = get_list_index(size);
        insert_to_list(bp, index);
    } else {                                   /* Case 4 */
        size_t prev_size = GET_SIZE(HDRP(PREV_BLKP(bp)));
        size_t next_size = GET_SIZE(HDRP(NEXT_BLKP(bp)));
        int prev_idx = get_list_index(prev_size);
        int next_idx = get_list_index(next_size);
        size += prev_size + next_size;
        remove_from_list(PREV_BLKP(bp), prev_idx);
        remove_from_list(NEXT_BLKP(bp), next_idx);
        PUT(HDRP(PREV_BLKP(bp)), PACK(size, 0));
        PUT(FTRP(NEXT_BLKP(bp)), PACK(size, 0));
        bp = PREV_BLKP(bp);
        int index = get_list_index(size);
        insert_to_list(bp, index);
    }
    return bp;
}

/*
 * find_fit - Find a fit for a block with asize bytes.
 * Uses first-fit for quick-lists, limited best-fit for segregated lists.
 */
static void *find_fit(size_t asize) {
    int i;
    char *bp;
    int start_index = get_list_index(asize);

    /* For quick-fit sizes, check exact size list first */
    if (asize <= QUICK_MAX_SIZE) {
        int quick_idx = QUICK_INDEX(asize);
        if (quick_idx < NUM_QUICK_LISTS) {
            /* First check exact size quick-fit list - first-fit for speed */
            if (free_lists[quick_idx] != NULL) {
                return free_lists[quick_idx];
            }
            /* Fall through to check larger quick-fit lists */
            start_index = quick_idx + 1;
        }
    }

    /* Search through lists from smallest appropriate class up */
    for (i = start_index; i < NUM_LISTS; i++) {
        char *best_bp = NULL;
        size_t best_size = (size_t)-1;
        int scan_count = 0;

        /* Best-fit search within this list, limited to 24 blocks for balance */
        for (bp = free_lists[i]; bp != NULL && scan_count < 24; bp = SUCC(bp), scan_count++) {
            size_t bsize = GET_SIZE(HDRP(bp));
            if (asize <= bsize && bsize < best_size) {
                best_bp = bp;
                best_size = bsize;
                /* Exact fit - return immediately */
                if (bsize == asize) {
                    return bp;
                }
            }
        }
        if (best_bp != NULL) {
            return best_bp;
        }
    }
    return NULL; /* No fit */
}

/*
 * place - Place block of asize bytes at start of free block bp and split
 *         if remainder would be at least SPLIT_THRESHOLD bytes.
 */
static void place(void *bp, size_t asize) {
    size_t csize = GET_SIZE(HDRP(bp));
    int index = get_list_index(csize);

    remove_from_list(bp, index);

    if ((csize - asize) >= SPLIT_THRESH) {
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        bp = NEXT_BLKP(bp);
        PUT(HDRP(bp), PACK(csize - asize, 0));
        PUT(FTRP(bp), PACK(csize - asize, 0));
        int new_index = get_list_index(csize - asize);
        insert_to_list(bp, new_index);
    } else {
        PUT(HDRP(bp), PACK(csize, 1));
        PUT(FTRP(bp), PACK(csize, 1));
    }
}

/*
 * mm_malloc - Allocate a block with at least size bytes of payload.
 */
void *mm_malloc(size_t size) {
    size_t asize;      /* Adjusted block size */
    size_t extendsize; /* Amount to extend heap if no fit */
    char *bp;

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs */
    if (size <= DSIZE)
        asize = MIN_BLOCK;
    else
        asize = DSIZE * ((size + (DSIZE) + (DSIZE - 1)) / DSIZE);

    /* Ensure minimum block size for free list pointers */
    if (asize < MIN_BLOCK)
        asize = MIN_BLOCK;

    /* Round up to quick-fit size class if applicable */
    if (asize <= QUICK_MAX_SIZE) {
        asize = ((asize - MIN_BLOCK + 15) / 16) * 16 + MIN_BLOCK;
        if (asize > QUICK_MAX_SIZE) asize = QUICK_MAX_SIZE;
    }

    /* Search the free list for a fit */
    if ((bp = find_fit(asize)) != NULL) {
        place(bp, asize);
        return bp;
    }

    /* No fit found. Get more memory and place the block */
    extendsize = MAX(asize, CHUNKSIZE);
    if ((bp = extend_heap(extendsize / WSIZE)) == NULL)
        return NULL;
    place(bp, asize);
    return bp;
}

/*
 * mm_free - Free a block and coalesce with neighbors.
 */
void mm_free(void *bp) {
    size_t size = GET_SIZE(HDRP(bp));

    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    coalesce(bp);
}

/*
 * mm_realloc - Reallocate a block with optimization for same-size and growth cases.
 */
void *mm_realloc(void *ptr, size_t size) {
    size_t oldsize;
    void *newptr;
    size_t asize;

    /* If size == 0 then this is just free, and we return NULL. */
    if (size == 0) {
        mm_free(ptr);
        return NULL;
    }

    /* If oldptr is NULL, then this is just malloc. */
    if (ptr == NULL) {
        return mm_malloc(size);
    }

    /* Calculate adjusted size */
    if (size <= DSIZE)
        asize = MIN_BLOCK;
    else
        asize = DSIZE * ((size + (DSIZE) + (DSIZE - 1)) / DSIZE);
    if (asize < MIN_BLOCK)
        asize = MIN_BLOCK;

    /* Round up to quick-fit size class if applicable */
    if (asize <= QUICK_MAX_SIZE) {
        asize = ((asize - MIN_BLOCK + 15) / 16) * 16 + MIN_BLOCK;
        if (asize > QUICK_MAX_SIZE) asize = QUICK_MAX_SIZE;
    }

    oldsize = GET_SIZE(HDRP(ptr));

    /* If new size is same or smaller, just return the original block */
    if (asize <= oldsize) {
        /* Optionally split if there's enough space */
        if (oldsize - asize >= SPLIT_THRESH) {
            PUT(HDRP(ptr), PACK(asize, 1));
            PUT(FTRP(ptr), PACK(asize, 1));
            newptr = NEXT_BLKP(ptr);
            PUT(HDRP(newptr), PACK(oldsize - asize, 0));
            PUT(FTRP(newptr), PACK(oldsize - asize, 0));
            int index = get_list_index(oldsize - asize);
            insert_to_list(newptr, index);
        }
        return ptr;
    }

    /* Check if next block is free and can be combined */
    char *next_bp = NEXT_BLKP(ptr);
    size_t next_size = GET_SIZE(HDRP(next_bp));
    size_t next_alloc = GET_ALLOC(HDRP(next_bp));

    if (!next_alloc && (oldsize + next_size >= asize)) {
        /* Combine with next block */
        int next_idx = get_list_index(next_size);
        remove_from_list(next_bp, next_idx);
        size_t combined_size = oldsize + next_size;

        if (combined_size - asize >= SPLIT_THRESH) {
            PUT(HDRP(ptr), PACK(asize, 1));
            PUT(FTRP(ptr), PACK(asize, 1));
            newptr = NEXT_BLKP(ptr);
            PUT(HDRP(newptr), PACK(combined_size - asize, 0));
            PUT(FTRP(newptr), PACK(combined_size - asize, 0));
            int index = get_list_index(combined_size - asize);
            insert_to_list(newptr, index);
        } else {
            PUT(HDRP(ptr), PACK(combined_size, 1));
            PUT(FTRP(ptr), PACK(combined_size, 1));
        }
        return ptr;
    }

    /* Check if previous block is free and can be combined */
    char *prev_bp = PREV_BLKP(ptr);
    size_t prev_size = GET_SIZE(HDRP(prev_bp));
    size_t prev_alloc = GET_ALLOC(FTRP(prev_bp));

    if (!prev_alloc && (oldsize + prev_size >= asize)) {
        /* Combine with previous block - need to move data */
        int prev_idx = get_list_index(prev_size);
        remove_from_list(prev_bp, prev_idx);
        size_t combined_size = oldsize + prev_size;

        PUT(HDRP(prev_bp), PACK(combined_size, 1));
        PUT(FTRP(prev_bp), PACK(combined_size, 1));

        /* Copy data to new location */
        memmove(prev_bp, ptr, oldsize - DSIZE);

        return prev_bp;
    }

    /* Check if both prev and next are free and combined would work */
    if (!prev_alloc && !next_alloc && (oldsize + prev_size + next_size >= asize)) {
        int prev_idx = get_list_index(prev_size);
        int next_idx = get_list_index(next_size);
        remove_from_list(prev_bp, prev_idx);
        remove_from_list(next_bp, next_idx);
        size_t combined_size = oldsize + prev_size + next_size;

        PUT(HDRP(prev_bp), PACK(combined_size, 1));
        PUT(FTRP(next_bp), PACK(combined_size, 1));

        /* Copy data to new location */
        memmove(prev_bp, ptr, oldsize - DSIZE);

        return prev_bp;
    }

    /* Need to allocate new block */
    newptr = mm_malloc(size);
    if (newptr == NULL) {
        return NULL;
    }

    /* Copy the old data */
    if (size < oldsize)
        oldsize = size;
    memcpy(newptr, ptr, oldsize);

    /* Free the old block */
    mm_free(ptr);

    return newptr;
}

// EVOLVE-BLOCK-END
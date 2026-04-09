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

team_t team = {
    "ateam",
    "Harry Bovik",
    "bovik@cs.cmu.edu",
    "",
    ""};

/* 16-byte alignment */
#define ALIGNMENT 16
#define ALIGN(size) (((size) + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1))

/* Word and double-word sizes */
#define WSIZE 8
#define DSIZE 16
#define CHUNKSIZE 4096

/* Number of segregated free list classes */
#define NUM_CLASSES 20

/*
 * We use bit 0 for alloc, bit 1 for prev_alloc.
 * Free blocks have: header (8) + next_ptr (8) + prev_ptr (8) + footer (8) = 32 min
 * Allocated blocks have: header (8) + payload (no footer needed) = min payload 16 => 24 total
 * But we need 16-byte alignment for payload, so min block = 16 (header) + 16 (payload) = nah
 * Actually: header is 8 bytes, payload starts at bp. bp must be 16-aligned.
 * So header at bp-8. Min block with free list ptrs: 8(hdr) + 8(next) + 8(prev) + 8(ftr) = 32
 * Min alloc block: 8(hdr) + 16(payload, aligned) = 24, but 24 is not 16-aligned for next block.
 * Next block bp = bp + block_size, block_size must be multiple of 16.
 * So min block = 16 for alloc? header(8) + payload(8) = 16. But then free block needs 32.
 * We need min block = 32 to hold free list pointers and footer.
 * Actually let's keep MIN_BLOCK_SIZE = 32 for simplicity with the prev_alloc optimization.
 */

#define MIN_BLOCK_SIZE 32

#define MAX(x, y) ((x) > (y) ? (x) : (y))

/* Bit masks */
#define ALLOC_BIT  0x1UL
#define PALLOC_BIT 0x2UL  /* Previous block is allocated */
#define SIZE_MASK  (~0xFUL)

/* Pack size, alloc, and prev_alloc into a word */
#define PACK(size, alloc) ((size) | (alloc))
#define PACK3(size, alloc, palloc) ((size) | (alloc) | (palloc))

/* Read and write a word at address p */
#define GET(p) (*(size_t *)(p))
#define PUT(p, val) (*(size_t *)(p) = (val))

/* Read fields from address p */
#define GET_SIZE(p) (GET(p) & SIZE_MASK)
#define GET_ALLOC(p) (GET(p) & ALLOC_BIT)
#define GET_PALLOC(p) (GET(p) & PALLOC_BIT)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp) ((char *)(bp) - WSIZE)
/* Footer only valid for free blocks */
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)))
/* PREV_BLKP only valid when prev block is free (has footer) */
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE(((char *)(bp) - DSIZE)))

/* Free list pointers stored in payload area */
#define NEXT_FREE(bp) (*(char **)(bp))
#define PREV_FREE(bp) (*(char **)((char *)(bp) + WSIZE))

/* Global variables */
static char *heap_listp = NULL;
static char *free_lists[NUM_CLASSES];
static char *last_placed = NULL;

/* Get the segregated list index for a given size */
static inline int get_class(size_t size) {
    if (size <= 32) return 0;
    if (size <= 48) return 1;
    if (size <= 64) return 2;
    if (size <= 96) return 3;
    if (size <= 128) return 4;
    if (size <= 192) return 5;
    if (size <= 256) return 6;
    if (size <= 384) return 7;
    if (size <= 512) return 8;
    if (size <= 1024) return 9;
    if (size <= 2048) return 10;
    if (size <= 4096) return 11;
    if (size <= 8192) return 12;
    if (size <= 16384) return 13;
    if (size <= 32768) return 14;
    if (size <= 65536) return 15;
    if (size <= 131072) return 16;
    if (size <= 262144) return 17;
    if (size <= 524288) return 18;
    return 19;
}

/* Insert block bp into the appropriate free list (address-ordered for small, LIFO for large) */
static inline void insert_free_block(char *bp) {
    size_t size = GET_SIZE(HDRP(bp));
    int idx = get_class(size);
    char *head = free_lists[idx];

    /* LIFO insertion */
    NEXT_FREE(bp) = head;
    PREV_FREE(bp) = NULL;
    if (head != NULL) {
        PREV_FREE(head) = bp;
    }
    free_lists[idx] = bp;
}

/* Remove block bp from its free list */
static inline void remove_free_block(char *bp) {
    size_t size = GET_SIZE(HDRP(bp));
    int idx = get_class(size);
    char *prev = PREV_FREE(bp);
    char *next = NEXT_FREE(bp);

    if (prev != NULL) {
        NEXT_FREE(prev) = next;
    } else {
        free_lists[idx] = next;
    }
    if (next != NULL) {
        PREV_FREE(next) = prev;
    }
}

/* Set the prev_alloc bit of the next block */
static inline void set_next_palloc(char *bp, int palloc) {
    char *next = NEXT_BLKP(bp);
    size_t hdr = GET(HDRP(next));
    if (palloc)
        PUT(HDRP(next), hdr | PALLOC_BIT);
    else
        PUT(HDRP(next), hdr & ~PALLOC_BIT);
    /* If next block is free, also update its footer */
    if (!GET_ALLOC(HDRP(next)) && GET_SIZE(HDRP(next)) > 0) {
        PUT(FTRP(next), GET(HDRP(next)));
    }
}

/* Coalesce adjacent free blocks */
static inline char *coalesce(char *bp) {
    size_t prev_alloc = GET_PALLOC(HDRP(bp));  /* bit 1 set means prev is allocated */
    size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    size_t size = GET_SIZE(HDRP(bp));

    if (prev_alloc && next_alloc) {
        /* No coalescing needed */
    } else if (prev_alloc && !next_alloc) {
        char *next = NEXT_BLKP(bp);
        remove_free_block(next);
        size += GET_SIZE(HDRP(next));
        PUT(HDRP(bp), PACK3(size, 0, PALLOC_BIT));
        PUT(FTRP(bp), PACK3(size, 0, PALLOC_BIT));
    } else if (!prev_alloc && next_alloc) {
        char *prev = PREV_BLKP(bp);
        remove_free_block(prev);
        size += GET_SIZE(HDRP(prev));
        size_t prev_palloc = GET_PALLOC(HDRP(prev));
        PUT(HDRP(prev), PACK3(size, 0, prev_palloc));
        PUT(FTRP(prev), PACK3(size, 0, prev_palloc));
        bp = prev;
    } else {
        char *prev = PREV_BLKP(bp);
        char *next = NEXT_BLKP(bp);
        remove_free_block(prev);
        remove_free_block(next);
        size += GET_SIZE(HDRP(prev)) + GET_SIZE(HDRP(next));
        size_t prev_palloc = GET_PALLOC(HDRP(prev));
        PUT(HDRP(prev), PACK3(size, 0, prev_palloc));
        PUT(FTRP(prev), PACK3(size, 0, prev_palloc));
        bp = prev;
    }

    /* Update next block's prev_alloc bit to 0 (we are free) */
    set_next_palloc(bp, 0);

    insert_free_block(bp);
    return bp;
}

/* Extend the heap by 'words' bytes */
static char *extend_heap(size_t words) {
    size_t size = ALIGN(words);
    char *bp = mem_sbrk(size);
    if (bp == (void *)-1)
        return NULL;

    /* Preserve the prev_alloc bit from the old epilogue */
    size_t old_palloc = GET_PALLOC(HDRP(bp));

    /* Initialize free block header/footer and the epilogue header */
    PUT(HDRP(bp), PACK3(size, 0, old_palloc));
    PUT(FTRP(bp), PACK3(size, 0, old_palloc));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* New epilogue header */

    return coalesce(bp);
}

/* Find a fit in the segregated free lists - best fit with limit */
static inline char *find_fit(size_t asize) {
    int idx = get_class(asize);

    for (int i = idx; i < NUM_CLASSES; i++) {
        char *bp = free_lists[i];
        char *best = NULL;
        size_t best_size = ~0UL;
        int count = 0;
        int limit = (i == idx) ? 30 : 3; /* search more in exact class */

        while (bp != NULL && count < limit) {
            size_t bsize = GET_SIZE(HDRP(bp));
            if (bsize >= asize) {
                if (bsize == asize) return bp; /* exact fit */
                if (bsize < best_size) {
                    best_size = bsize;
                    best = bp;
                    if (bsize < asize + 32) break; /* close enough */
                }
            }
            bp = NEXT_FREE(bp);
            count++;
        }
        if (best != NULL) return best;
    }
    return NULL;
}

/* Place a block of asize bytes at bp, splitting if remainder is large enough */
static inline void place(char *bp, size_t asize) {
    size_t csize = GET_SIZE(HDRP(bp));
    size_t palloc = GET_PALLOC(HDRP(bp));
    size_t remainder = csize - asize;
    remove_free_block(bp);

    if (remainder >= MIN_BLOCK_SIZE) {
        /*
         * For smaller allocations (asize <= 128), place the allocated block
         * at the END of the free block. This helps binary traces where two
         * sizes alternate: the front remainder stays at the same address
         * and can be reused, reducing fragmentation.
         * For larger allocations, place at the front (normal strategy) to
         * avoid unnecessary splitting overhead and help realloc expand forward.
         */
        if (asize >= 96) {
            /* Free block at front, allocated block at back */
            PUT(HDRP(bp), PACK3(remainder, 0, palloc));
            PUT(FTRP(bp), PACK3(remainder, 0, palloc));
            char *alloc_bp = NEXT_BLKP(bp);
            PUT(HDRP(alloc_bp), PACK3(asize, 1, 0)); /* prev is free */
            /* Update the block after the allocated block */
            set_next_palloc(alloc_bp, 1);
            insert_free_block(bp);
            /* Note: caller will use the return value from find_fit,
             * but we changed where the allocation is. We need to return
             * the new bp. We can't do that from place() directly.
             * So we use a global to communicate this. */
            last_placed = alloc_bp;
        } else {
            /* Normal: allocated block at front, free remainder at back */
            PUT(HDRP(bp), PACK3(asize, 1, palloc));
            char *next = NEXT_BLKP(bp);
            PUT(HDRP(next), PACK3(remainder, 0, PALLOC_BIT));
            PUT(FTRP(next), PACK3(remainder, 0, PALLOC_BIT));
            set_next_palloc(next, 0);
            insert_free_block(next);
            last_placed = bp;
        }
    } else {
        PUT(HDRP(bp), PACK3(csize, 1, palloc));
        set_next_palloc(bp, 1);
        last_placed = bp;
    }
}

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) {
    /* Initialize free lists */
    for (int i = 0; i < NUM_CLASSES; i++) {
        free_lists[i] = NULL;
    }

    /* Create the initial empty heap */
    char *p = mem_sbrk(4 * WSIZE);
    if (p == (void *)-1)
        return -1;

    PUT(p, 0);                                        /* Alignment padding */
    PUT(p + (1 * WSIZE), PACK3(DSIZE, 1, PALLOC_BIT)); /* Prologue header */
    PUT(p + (2 * WSIZE), PACK3(DSIZE, 1, PALLOC_BIT)); /* Prologue footer */
    PUT(p + (3 * WSIZE), PACK3(0, 1, PALLOC_BIT));     /* Epilogue header */
    heap_listp = p + (2 * WSIZE);

    /* Extend the empty heap with a small initial free block */
    if (extend_heap(256) == NULL)
        return -1;

    return 0;
}

/*
 * mm_malloc - Allocate a block with segregated free list.
 */
void *mm_malloc(size_t size) {
    if (size == 0)
        return NULL;

    size_t asize;
    if (size <= DSIZE + WSIZE)
        asize = MIN_BLOCK_SIZE;
    else
        asize = ALIGN(size + WSIZE); /* only header overhead, no footer for alloc */

    /* Search the free lists for a fit */
    char *bp = find_fit(asize);
    if (bp != NULL) {
        place(bp, asize);
        return last_placed;
    }

    /* No fit found. Get more memory - adaptive extend size */
    size_t extendsize;
    if (asize >= 4096)
        extendsize = asize;
    else if (asize >= 512)
        extendsize = MAX(asize, 4096);
    else if (asize >= 96)
        extendsize = MAX(asize, 2048);
    else
        extendsize = MAX(asize, CHUNKSIZE);
    bp = extend_heap(extendsize);
    if (bp == NULL)
        return NULL;
    place(bp, asize);
    return last_placed;
}

/*
 * mm_free - Free a block and coalesce.
 */
void mm_free(void *ptr) {
    if (ptr == NULL)
        return;

    size_t size = GET_SIZE(HDRP(ptr));
    size_t palloc = GET_PALLOC(HDRP(ptr));
    PUT(HDRP(ptr), PACK3(size, 0, palloc));
    PUT(FTRP(ptr), PACK3(size, 0, palloc));
    coalesce(ptr);
}

/*
 * mm_realloc - Reallocate with in-place expansion when possible.
 */
void *mm_realloc(void *ptr, size_t size) {
    if (ptr == NULL)
        return mm_malloc(size);
    if (size == 0) {
        mm_free(ptr);
        return NULL;
    }

    size_t asize;
    if (size <= DSIZE + WSIZE)
        asize = MIN_BLOCK_SIZE;
    else
        asize = ALIGN(size + WSIZE);

    size_t oldsize = GET_SIZE(HDRP(ptr));

    /* If current block is large enough */
    if (oldsize >= asize) {
        /* Split if there's enough excess */
        if ((oldsize - asize) >= MIN_BLOCK_SIZE) {
            size_t palloc = GET_PALLOC(HDRP(ptr));
            PUT(HDRP(ptr), PACK3(asize, 1, palloc));
            char *next = NEXT_BLKP(ptr);
            PUT(HDRP(next), PACK3(oldsize - asize, 0, PALLOC_BIT));
            PUT(FTRP(next), PACK3(oldsize - asize, 0, PALLOC_BIT));
            /* Coalesce the remainder with whatever follows */
            coalesce(next);
        }
        return ptr;
    }

    /* Try to expand into the next block if it's free */
    char *next = NEXT_BLKP(ptr);
    size_t next_size = GET_SIZE(HDRP(next));
    int next_alloc = GET_ALLOC(HDRP(next));

    if (!next_alloc && (oldsize + next_size) >= asize) {
        remove_free_block(next);
        size_t combined = oldsize + next_size;
        size_t palloc = GET_PALLOC(HDRP(ptr));
        if ((combined - asize) >= MIN_BLOCK_SIZE) {
            PUT(HDRP(ptr), PACK3(asize, 1, palloc));
            char *split = NEXT_BLKP(ptr);
            PUT(HDRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
            PUT(FTRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
            set_next_palloc(split, 0);
            insert_free_block(split);
        } else {
            PUT(HDRP(ptr), PACK3(combined, 1, palloc));
            set_next_palloc(ptr, 1);
        }
        return ptr;
    }

    /* If next block is the epilogue (end of heap), extend */
    if (next_size == 0) {
        size_t extend = asize - oldsize;
        extend = ALIGN(extend);
        char *bp = mem_sbrk(extend);
        if (bp == (void *)-1)
            goto try_prev;
        size_t combined = oldsize + extend;
        size_t palloc = GET_PALLOC(HDRP(ptr));
        PUT(HDRP(ptr), PACK3(combined, 1, palloc));
        PUT(HDRP(NEXT_BLKP(ptr)), PACK3(0, 1, PALLOC_BIT)); /* New epilogue */
        /* Split if there's excess */
        if ((combined - asize) >= MIN_BLOCK_SIZE) {
            PUT(HDRP(ptr), PACK3(asize, 1, palloc));
            char *split = NEXT_BLKP(ptr);
            PUT(HDRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
            PUT(FTRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
            set_next_palloc(split, 0);
            insert_free_block(split);
        }
        return ptr;
    }

    /* Try: next is free and after next is epilogue */
    if (!next_alloc) {
        char *after_next = NEXT_BLKP(next);
        size_t after_next_size = GET_SIZE(HDRP(after_next));
        if (after_next_size == 0) {
            remove_free_block(next);
            size_t combined = oldsize + next_size;
            size_t extend = 0;
            if (asize > combined) {
                extend = asize - combined;
                char *bp = mem_sbrk(extend);
                if (bp == (void *)-1) {
                    insert_free_block(next);
                    goto try_prev;
                }
                combined += extend;
            }
            size_t palloc = GET_PALLOC(HDRP(ptr));
            PUT(HDRP(ptr), PACK3(combined, 1, palloc));
            PUT(HDRP(NEXT_BLKP(ptr)), PACK3(0, 1, PALLOC_BIT));
            if ((combined - asize) >= MIN_BLOCK_SIZE) {
                PUT(HDRP(ptr), PACK3(asize, 1, palloc));
                char *split = NEXT_BLKP(ptr);
                PUT(HDRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
                PUT(FTRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
                set_next_palloc(split, 0);
                insert_free_block(split);
            }
            return ptr;
        }
    }

try_prev:;
    /* Try to expand backward into previous free block */
    if (!GET_PALLOC(HDRP(ptr))) {
        char *prev = PREV_BLKP(ptr);
        size_t prev_size = GET_SIZE(HDRP(prev));
        size_t combined = oldsize + prev_size;

        /* Also check if next is free */
        next = NEXT_BLKP(ptr);
        next_size = GET_SIZE(HDRP(next));
        next_alloc = GET_ALLOC(HDRP(next));

        if (!next_alloc) {
            combined += next_size;
        }

        if (combined >= asize) {
            remove_free_block(prev);
            if (!next_alloc) {
                remove_free_block(next);
            }
            size_t prev_palloc = GET_PALLOC(HDRP(prev));
            /* Move data to prev */
            memmove(prev, ptr, oldsize - WSIZE);
            if ((combined - asize) >= MIN_BLOCK_SIZE) {
                PUT(HDRP(prev), PACK3(asize, 1, prev_palloc));
                char *split = NEXT_BLKP(prev);
                PUT(HDRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
                PUT(FTRP(split), PACK3(combined - asize, 0, PALLOC_BIT));
                set_next_palloc(split, 0);
                insert_free_block(split);
            } else {
                PUT(HDRP(prev), PACK3(combined, 1, prev_palloc));
                set_next_palloc(prev, 1);
            }
            return prev;
        }
    }

    /* Fall back to malloc + copy + free */
    void *newptr = mm_malloc(size);
    if (newptr == NULL)
        return NULL;
    size_t copySize = oldsize - WSIZE; /* payload of old block (only header overhead) */
    if (size < copySize)
        copySize = size;
    memcpy(newptr, ptr, copySize);
    mm_free(ptr);
    return newptr;
}

// EVOLVE-BLOCK-END
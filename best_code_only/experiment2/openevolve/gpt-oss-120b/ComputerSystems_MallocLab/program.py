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


/* Minimum block size that can remain after a split (header + alignment) */
#define MIN_BLOCK_SIZE (ALIGN(sizeof(header_t) + ALIGNMENT))
/* Allocate heap extensions in 4 KB chunks to amortize sbrk calls */
#define CHUNK_SIZE (1 << 12)   /* 4096 bytes */

/*-------------------------------------------------
 * Real free‑list implementation.
 * Each block starts with a header that stores the total
 * block size (including the header) and a pointer to the
 * next free block.  The header is 16‑byte aligned on
 * 64‑bit systems, satisfying the required ALIGNMENT.
 *-------------------------------------------------*/
typedef struct header {
    size_t size;          /* total size of the block (header+payload) */
    struct header *next;  /* next free block (NULL if none) */
} header_t;

/* head of the singly‑linked free list */
static header_t *free_list_head = NULL;
/* cache of the most‑recently freed block – used as a fast‑path in mm_malloc */
static header_t *last_fit = NULL;

/* Forward declaration of the coalescing helper */
/* The explicit free list is now kept in address order.
   Coalescing is handled directly inside mm_free, so the
   separate coalesce helper is no longer needed. */

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) {
    /* reset the free‑list and fast‑path cache for each new trace */
    free_list_head = NULL;
    last_fit = NULL;            /* <-- reset cached block */
    return 0;
}

/*
 * mm_malloc - Allocate a block by incrementing the brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size) {
    /* Conform to C standard: malloc(0) should return NULL */
    if (size == 0)
        return NULL;
    /* total block size needed (header + payload, aligned) */
    size_t asize = ALIGN(size + sizeof(header_t));

    /* ---------- Fast path: reuse the most‑recently freed block if it fits ---------- */
    if (last_fit && last_fit->size >= asize) {
        /* Fast‑path: the cached block is guaranteed to be in the free list */
        header_t *block = last_fit;
        /* Remove it from the free list */
        header_t **link = &free_list_head;
        while (*link && *link != block) {
            link = &(*link)->next;
        }
        if (*link) {
            *link = block->next;      /* unlink */
            last_fit = NULL;           /* cache consumed */

            size_t excess = block->size - asize;
            if (excess >= MIN_BLOCK_SIZE) {
                header_t *new_free = (header_t *)((char *)block + asize);
                new_free->size = excess;

                /* Insert the remainder back into the free list (sorted by address) */
                header_t **ins = &free_list_head;
                while (*ins && *ins < new_free) {
                    ins = &(*ins)->next;
                }
                new_free->next = *ins;
                *ins = new_free;

                block->size = asize;
            }
            return (void *)((char *)block + sizeof(header_t));
        }
    }

    /* ---------- 1. Look for a fitting free block ---------- */
    header_t **prev = &free_list_head;
    header_t *curr = free_list_head;
    while (curr != NULL) {
        if (curr->size >= asize) {
            /* remove from free list */
            *prev = curr->next;

            /* ---------- optional split ---------- */
            size_t excess = curr->size - asize;
            /* If the remaining part of the block is large enough to be a free
               block, split it and insert the remainder back into the free list.
               The free list is kept **sorted by address** so that later
               coalescing (in mm_free) can merge adjacent blocks correctly. */
            if (excess >= MIN_BLOCK_SIZE) {
                header_t *new_free = (header_t *)((char *)curr + asize);
                new_free->size = excess;

                /* Insert new_free into the free list in address order */
                header_t **link = &free_list_head;
                while (*link && *link < new_free) {
                    link = &(*link)->next;
                }
                new_free->next = *link;
                *link = new_free;

                curr->size = asize;
            }

            /* payload starts just after the header */
            return (void *)((char *)curr + sizeof(header_t));
        }
        prev = &curr->next;
        curr = curr->next;
    }

    /* ---------- 2. No fit – extend the heap ---------- */
    /* Allocate in 4 KB chunks; any leftover after satisfying the request
       is placed back onto the free list. */
    size_t extend = (asize > CHUNK_SIZE) ? asize : CHUNK_SIZE;
    void *p = mem_sbrk(extend);
    if (p == (void *)-1)
        return NULL;

    header_t *hdr = (header_t *)p;
    hdr->size = asize;
    hdr->next = NULL;               /* not used for allocated blocks */

    /* No free block was found, so the fast‑path cache must be empty */
    last_fit = NULL;                /* <-- ensure cache is cleared */

    return (void *)((char *)hdr + sizeof(header_t));
}

/*
 * mm_free - Freeing a block does nothing.
 */
void mm_free(void *ptr) {
    if (ptr == NULL)
        return;

    /* Obtain the block header (located just before the payload) */
    header_t *hdr = (header_t *)ptr - 1;

    /* Insert the freed block into the free list in address order */
    header_t **link = &free_list_head;
    header_t *prev_block = NULL;
    while (*link && *link < hdr) {
        prev_block = *link;
        link = &(*link)->next;
    }
    hdr->next = *link;
    *link = hdr;
    /* Update the fast‑path cache with this newly freed block */
    last_fit = hdr;

    /* Coalesce with next block if adjacent */
    if (hdr->next && (char *)hdr + hdr->size == (char *)hdr->next) {
        hdr->size += hdr->next->size;
        hdr->next = hdr->next->next;
    }

    /* Coalesce with previous block if adjacent */
    if (prev_block && (char *)prev_block + prev_block->size == (char *)hdr) {
        prev_block->size += hdr->size;
        prev_block->next = hdr->next;
    }
}

/* -----------------------------------------------------------
 *  coalesce – merge a freshly freed block with any neighboring
 *  free blocks that are contiguous in memory.  This keeps the
 *  free list from becoming highly fragmented and prevents
 *  premature out‑of‑memory failures.
 * ----------------------------------------------------------- */
/* The old coalesce helper has been removed – its logic is now
   integrated into mm_free for O(1) adjacency merges. */

/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
void *mm_realloc(void *ptr, size_t size) {
    /* Handle the two degenerate cases first */
    if (ptr == NULL)
        return mm_malloc(size);
    if (size == 0) {
        mm_free(ptr);
        return NULL;
    }

    /* Locate the old block's header */
    header_t *old_hdr = (header_t *)ptr - 1;
    size_t old_payload = old_hdr->size - sizeof(header_t);

    /* Fast path: request fits in the existing block */
    if (size <= old_payload)
        return ptr;

    /* Try to extend the block in‑place by merging with an adjacent free block */
    size_t needed = ALIGN(size + sizeof(header_t));
    header_t **link = &free_list_head;
    while (*link) {
        /* Is the free block directly after the current one? */
        if ((char *)(*link) == (char *)old_hdr + old_hdr->size) {
            size_t combined = old_hdr->size + (*link)->size;
            if (combined >= needed) {
                /* Remove the adjacent free block from the free list */
                header_t *adj = *link;
                *link = adj->next;

                old_hdr->size = combined;

                /* Split the enlarged block if there is excess space */
                size_t excess = old_hdr->size - needed;
                if (excess >= MIN_BLOCK_SIZE) {
                    header_t *split = (header_t *)((char *)old_hdr + needed);
                    split->size = excess;
                    /* Insert the split block back into the free list (sorted) */
                    header_t **slink = &free_list_head;
                    while (*slink && *slink < split) {
                        slink = &(*slink)->next;
                    }
                    split->next = *slink;
                    *slink = split;
                    old_hdr->size = needed;
                }
                /* Return the same payload pointer */
                return (void *)((char *)old_hdr + sizeof(header_t));
            }
            /* Adjacent free block not big enough – cannot extend */
            break;
        }
        link = &(*link)->next;
    }

    /* Fallback: allocate a new block, copy data, free old block */
    void *newptr = mm_malloc(size);
    if (newptr == NULL)
        return NULL;

    size_t copySize = (old_payload < size) ? old_payload : size;
    memcpy(newptr, ptr, copySize);
    mm_free(ptr);
    return newptr;
}

// EVOLVE-BLOCK-END

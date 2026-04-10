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
#define ALIGN(size) (((size)+(ALIGNMENT-1))&~(ALIGNMENT-1))
#define WSIZE sizeof(size_t)
#define DSIZE (2*WSIZE)
#define OVERHEAD (2*WSIZE)
#define CHUNKSIZE (1<<12)
#define PACK(size,alloc) ((size)|(alloc))
#define GET(p) (*(size_t *)(p))
#define PUT(p,val) (*(size_t *)(p)=(val))
#define GET_SIZE(p) (GET(p)&~(size_t)0xF)
#define GET_ALLOC(p) (GET(p)&0x1)
#define HDRP(bp) ((char *)(bp)-WSIZE)
#define FTRP(bp) ((char *)(bp)+GET_SIZE(HDRP(bp))-DSIZE)
#define NEXT_BLKP(bp) ((char *)(bp)+GET_SIZE(HDRP(bp)))
#define PREV_FTRP(bp) ((char *)(bp)-DSIZE)
#define PREV_BLKP(bp) ((char *)(bp)-GET_SIZE(PREV_FTRP(bp)))
#define MINBLOCK (2*DSIZE)

static char *heap_listp,*last_bp;

static void *coalesce(void *bp){
  size_t prev_alloc=GET_ALLOC(PREV_FTRP(bp)),next_alloc=GET_ALLOC(HDRP(NEXT_BLKP(bp))),size=GET_SIZE(HDRP(bp));
  if(prev_alloc&&next_alloc) return bp;
  if(prev_alloc&&!next_alloc){
    size+=GET_SIZE(HDRP(NEXT_BLKP(bp)));
    PUT(HDRP(bp),PACK(size,0)); PUT(FTRP(bp),PACK(size,0));
    return bp;
  }
  if(!prev_alloc&&next_alloc){
    bp=PREV_BLKP(bp); size+=GET_SIZE(HDRP(bp));
    PUT(HDRP(bp),PACK(size,0)); PUT(FTRP(bp),PACK(size,0));
    return bp;
  }
  size+=GET_SIZE(HDRP(PREV_BLKP(bp)))+GET_SIZE(HDRP(NEXT_BLKP(bp)));
  PUT(HDRP(PREV_BLKP(bp)),PACK(size,0)); PUT(FTRP(NEXT_BLKP(bp)),PACK(size,0));
  return PREV_BLKP(bp);
}

static void *extend_heap(size_t bytes){
  size_t size=ALIGN(bytes); char *bp=mem_sbrk(size);
  if(bp==(void *)-1) return NULL;
  PUT(HDRP(bp),PACK(size,0)); PUT(FTRP(bp),PACK(size,0)); PUT(HDRP(NEXT_BLKP(bp)),PACK(0,1));
  return coalesce(bp);
}

static void place(void *bp,size_t asize){
  size_t csize=GET_SIZE(HDRP(bp));
  if(csize-asize>=MINBLOCK){
    PUT(HDRP(bp),PACK(asize,1)); PUT(FTRP(bp),PACK(asize,1));
    bp=NEXT_BLKP(bp);
    PUT(HDRP(bp),PACK(csize-asize,0)); PUT(FTRP(bp),PACK(csize-asize,0));
  }else{
    PUT(HDRP(bp),PACK(csize,1)); PUT(FTRP(bp),PACK(csize,1));
  }
}

static void *find_fit(size_t asize){
  char *bp=last_bp,*start=bp;
  for(;GET_SIZE(HDRP(bp))>0;bp=NEXT_BLKP(bp))
    if(!GET_ALLOC(HDRP(bp))&&GET_SIZE(HDRP(bp))>=asize) return last_bp=bp;
  for(bp=heap_listp;bp<start;bp=NEXT_BLKP(bp))
    if(!GET_ALLOC(HDRP(bp))&&GET_SIZE(HDRP(bp))>=asize) return last_bp=bp;
  return NULL;
}

int mm_init(void){
  char *p=mem_sbrk(4*WSIZE);
  if(p==(void *)-1) return -1;
  PUT(p,0); PUT(p+WSIZE,PACK(DSIZE,1)); PUT(p+2*WSIZE,PACK(DSIZE,1)); PUT(p+3*WSIZE,PACK(0,1));
  heap_listp=last_bp=p+2*WSIZE;
  return extend_heap(CHUNKSIZE)!=NULL?0:-1;
}

void *mm_malloc(size_t size){
  size_t asize,extendsize; char *bp;
  if(size==0) return NULL;
  asize=ALIGN(size+OVERHEAD); if(asize<MINBLOCK) asize=MINBLOCK;
  if((bp=find_fit(asize))!=NULL){place(bp,asize);return bp;}
  extendsize=asize>CHUNKSIZE?asize:CHUNKSIZE;
  if((bp=extend_heap(extendsize))==NULL) return NULL;
  place(bp,asize);
  return bp;
}

void mm_free(void *ptr){
  size_t size;
  if(!ptr) return;
  size=GET_SIZE(HDRP(ptr));
  PUT(HDRP(ptr),PACK(size,0)); PUT(FTRP(ptr),PACK(size,0));
  last_bp=coalesce(ptr);
}

void *mm_realloc(void *ptr,size_t size){
  size_t asize,oldsize,nextsize,copySize,total,need; void *newptr,*next;
  if(ptr==NULL) return mm_malloc(size);
  if(size==0){mm_free(ptr);return NULL;}
  asize=ALIGN(size+OVERHEAD); if(asize<MINBLOCK) asize=MINBLOCK;
  oldsize=GET_SIZE(HDRP(ptr));
  if(asize<=oldsize){
    if(oldsize-asize>=MINBLOCK) place(ptr,asize);
    return ptr;
  }
  next=NEXT_BLKP(ptr);
  nextsize=GET_SIZE(HDRP(next));
  if(!GET_ALLOC(HDRP(next))){
    total=oldsize+nextsize;
    if(total>=asize){
      PUT(HDRP(ptr),PACK(total,1)); PUT(FTRP(ptr),PACK(total,1));
      if(total-asize>=MINBLOCK) place(ptr,asize);
      return ptr;
    }
  }else if(nextsize==0){
    need=asize-oldsize;
    if(extend_heap(need>CHUNKSIZE?need:CHUNKSIZE)==next){
      total=oldsize+GET_SIZE(HDRP(next));
      PUT(HDRP(ptr),PACK(total,1)); PUT(FTRP(ptr),PACK(total,1));
      if(total-asize>=MINBLOCK) place(ptr,asize);
      return ptr;
    }
  }
  newptr=mm_malloc(size);
  if(newptr==NULL) return NULL;
  copySize=oldsize-OVERHEAD; if(size<copySize) copySize=size;
  memcpy(newptr,ptr,copySize);
  mm_free(ptr);
  return newptr;
}

// EVOLVE-BLOCK-END

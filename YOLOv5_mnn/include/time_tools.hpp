#include <stdio.h>
#include<time.h>

static int timespec_check(struct timespec *t) 
 { 
     if((t->tv_nsec <0 ) || (t->tv_nsec >= 1000000000)) 
         return -1; 
  
     return 0; 
 } 

static void timespec_sub(struct timespec *t1,  struct timespec *t2, struct timespec *time_) 
 { 
     if (timespec_check(t1) < 0) { 
         fprintf(stderr, "invalid time #1: %lld.%.9ld.\n", 
             (long long) t1->tv_sec,t1->tv_nsec); 
         return; 
     } 
     if (timespec_check(t2) < 0) { 
         fprintf(stderr, "invalid time #2: %lld.%.9ld.\n", 
             (long long) t2->tv_sec,t2->tv_nsec); 
         return; 
     } 
  
     time_->tv_sec = t1->tv_sec - t2->tv_sec; 
     time_->tv_nsec = t1->tv_nsec - t2->tv_nsec; 
     if (time_->tv_nsec >= 1000000000) 
     { 
         time_->tv_sec++; 
         time_->tv_nsec -= 1000000000; 
     } 
     else if (time_->tv_nsec < 0) 
     { 
         time_->tv_sec--; 
         time_->tv_nsec += 1000000000; 
     } 
} 
#include "param.h"
#include "types.h"
#include "stat.h"
#include "user.h"
#include "fs.h"
#include "fcntl.h"
#include "syscall.h"
#include "traps.h"
#include "memlayout.h"

#define TOTAL_MEMORY (2 << 20) + (1 << 18) + (1 << 17)

void mem(void) {
    void *m1 = 0, *m2, *start;
    uint cur = 0;
    uint count = 0;
    uint total_count;

    printf(1, "memtest: Starting memory test, trying to allocate %d bytes (%d KB)\n", TOTAL_MEMORY, TOTAL_MEMORY / 1024);

    // Initial allocation
    m1 = malloc(4096);
    if (m1 == 0) {
        printf(1, "memtest: Initial allocation of 4KB failed.\n");
        goto failed;
    }
    start = m1;
    cur += 4096;  // Count the first allocation.

    while (cur < TOTAL_MEMORY) {
        m2 = malloc(4096);
        if (m2 == 0) {
            printf(1, "memtest: Allocation of 4KB failed at %u KB.\n", cur / 1024);
            goto failed;
        }
        *(char**)m1 = m2; // Link the previous block to the new one.
        
        // Store the index in the memory area
        ((int*)m1)[2] = count++;
        m1 = m2; // Move the pointer to the newly allocated block.
        cur += 4096; // Update total bytes allocated
    }

    // Store the total count in the last allocated block for verification.
    ((int*)m1)[2] = count;
    total_count = count;

    printf(1, "memtest: Successfully allocated %u blocks of 4KB memory (%u KB total).\n", total_count, cur / 1024);

    // Verification loop
    count = 0;
    m1 = start;

    while (count != total_count) {
        if (((int*)m1)[2] != count) {
            printf(1, "memtest: Verification failed at block %u, expected count %u but found %u.\n", count, count, ((int*)m1)[2]);
            goto failed;
        }
        m1 = *(char**)m1; // Move to the next allocated block.
        count++;
    }

    printf(1, "memtest: Memory test passed.\n");
    exit();

failed:
    printf(1, "memtest: Memory test failed after allocating %u KB.\n", cur / 1024);
    exit();
}

int main(int argc, char *argv[]) {
    mem();
    exit(); 
    return 0;
}
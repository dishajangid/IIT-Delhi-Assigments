#include "types.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "mmu.h"
#include "proc.h"
#include "spinlock.h"
#include "sleeplock.h"
#include "fs.h"
#include "buf.h"
#include "x86.h"

// External variables
extern struct {
  struct spinlock lock;
  struct proc proc[NPROC];
} ptable;

extern struct superblock sb;  // Need access to superblock for swap area

// Number of swap slots as per assignment (800 slots)
#define NSWAPSLOTS 800

// Swap slot structure with only two fields as specified
struct swapslot {
  int page_perm;  // Permission of the swapped memory page
  int is_free;    // 1 if free, 0 if used
};

// Define PTE_SWAPPED flag to identify swapped pages
#define PTE_SWAPPED 0x800  // Use a free bit in the PTE

struct swapslot swapslots[NSWAPSLOTS];
struct spinlock swaplock;

// Adaptive page replacement parameters
int TH = 100;     // Threshold for free pages
int NPG = 4;      // Number of pages to swap out at a time
#define LIMIT 100 // Maximum number of pages to swap out at once

// Our own implementation of bget to avoid using the static one in bio.c
struct buf*
pageswap_bget(uint dev, uint blockno)
{
  struct buf *b;
  b = bread(dev, blockno);
  return b;
}

// Our own implementation of walkpgdir to avoid using the static one in vm.c
pte_t*
pageswap_walkpgdir(pde_t *pgdir, const void *va, int alloc)
{
  pde_t *pde;
  pte_t *pgtab;

  pde = &pgdir[PDX(va)];
  if(*pde & PTE_P){
    pgtab = (pte_t*)P2V(PTE_ADDR(*pde));
  } else {
    if(!alloc || (pgtab = (pte_t*)kalloc()) == 0)
      return 0;
    // Make sure all those PTE_P bits are zero.
    memset(pgtab, 0, PGSIZE);
    // The permissions here are overly generous, but they can be
    // reset by mappages.
    *pde = V2P(pgtab) | PTE_P | PTE_W | PTE_U;
  }
  return &pgtab[PTX(va)];
}

// Our own implementation of mappages to avoid using the static one in vm.c
int
pageswap_mappages(pde_t *pgdir, void *va, uint size, uint pa, int perm)
{
  char *a, *last;
  pte_t *pte;

  a = (char*)PGROUNDDOWN((uint)va);
  last = (char*)PGROUNDDOWN(((uint)va) + size - 1);
  for(;;){
    if((pte = pageswap_walkpgdir(pgdir, a, 1)) == 0)
      return -1;
    if(*pte & PTE_P)
      panic("remap");
    *pte = pa | perm | PTE_P;
    if(a == last)
      break;
    a += PGSIZE;
    pa += PGSIZE;
  }
  return 0;
}

// Initialize swap system
void 
swapinit(void) {
  int i;
  initlock(&swaplock, "swap");
  for(i = 0; i < NSWAPSLOTS; i++){
    swapslots[i].page_perm = 0;
    swapslots[i].is_free = 1; 
  }
}

// Find a free swap slot
int 
allocswapslot(void) {
  int i;
  acquire(&swaplock);
  
  for(i = 0; i < NSWAPSLOTS; i++) {
    if(swapslots[i].is_free) {
      swapslots[i].is_free = 0;
      release(&swaplock);
      return i;
    }
  }
  release(&swaplock);
  return -1;
}



// Free a swap slot
void
freeswapslot(int slotno)
{
  if(slotno < 0 || slotno >= NSWAPSLOTS) {
    cprintf("ERROR: freeswapslot: Invalid slot number %d\n", slotno);
    return;
  }
  acquire(&swaplock);
  // Check if slot is already free
  if(swapslots[slotno].is_free) {
    // cprintf("WARNING: freeswapslot: Slot %d is already free\n", slotno);
    release(&swaplock);
    return;
  }
  swapslots[slotno].is_free = 1;
  swapslots[slotno].page_perm = 0;
  release(&swaplock);
  // cprintf("freeswapslot: Freed swap slot %d\n", slotno);
}


// Get the disk block number for a specific swap slot
uint
swapslot_blockno(int slotno)
{
  if(slotno < 0 || slotno >= NSWAPSLOTS)
    panic("swapslot_blockno: invalid slot");
    
  uint blockno = sb.swapstart + (slotno * 8); // Each slot is 8 blocks (4096 bytes / 512 bytes)
  // cprintf("Swap slot %d mapped to block number: %d\n", slotno, blockno);
  return blockno;
}

// Write a page to swap space
int
swapout_page(char *page, int perm)
{
  int slotno;
  uint blockno;

  if(page == 0) {
    // cprintf("ERROR: swapout_page: Null page pointer\n");
    return -1;
  }
  
  slotno = allocswapslot();
  if(slotno < 0) {
    // cprintf("ERROR: swapout_page: Failed to allocate swap slot\n");
    return -1; // No free swap slot
  }
  blockno = swapslot_blockno(slotno);
  // Store page permission in the slot
  acquire(&swaplock);
  swapslots[slotno].page_perm = perm;
  release(&swaplock);
  // Write the page to disk (bypassing log)
  // cprintf("swapout_page: Writing page to slot %d at block %d with perm 0x%x\n", slotno, blockno, perm);
  swapwrite(1, page, blockno, 8);  // Device 1, 8 blocks for the page
  return slotno;
}

// Read a page from swap space
int
swapin_page(char *page, int slotno)
{
  uint blockno;
  int perm;
  
  if(slotno < 0 || slotno >= NSWAPSLOTS) {
    // cprintf("ERROR: swapin_page: Invalid slot number %d\n", slotno);
    return -1; // Invalid slot
  }
    
  acquire(&swaplock);
  if(swapslots[slotno].is_free) {
    release(&swaplock);
    // cprintf("ERROR: swapin_page: Slot %d is marked as free\n", slotno);
    return -1; // Slot is free
  }
  
  perm = swapslots[slotno].page_perm;
  release(&swaplock);
  blockno = swapslot_blockno(slotno);
  // Read the page from disk
  // cprintf("swapin_page: Reading page from slot %d at block %d with perm 0x%x\n", slotno, blockno, perm);
  swapread(1, page, blockno, 8);  // Device 1, 8 blocks for the page
  return perm; // Return the page permissions
}


// Bypass log for swap operations - writing to disk
// Bypass log for swap operations - writing to disk
// void
// swapwrite(uint dev, char *buf, uint block, uint count)
// {
//   struct buf *b;

//   if(buf == 0) {
//     cprintf("ERROR: swapwrite: Null buffer pointer\n");
//     return;
//   }
  
//   for(int i = 0; i < count; i++){
//     b = pageswap_bget(dev, block + i);
//     if(b == 0) {
//       cprintf("swapwrite: failed to get buffer for block %d\n", block + i);
//       return; // Exit if we can't get a buffer - prevent panic
//     }
//     memmove(b->data, buf + (i * BSIZE), BSIZE);
//     b->flags |= B_DIRTY;
//     brelse(b);
//   }
//   cprintf("Written %d blocks to swap device %d starting from block %d\n", count, dev, block);
// }

void
swapwrite(uint dev, char *buf, uint block, uint count)
{
  // Handle writes using direct I/O instead of the buffer cache
  int i;
  struct buf *b;
  
  for(i = 0; i < count; i++){
    // Release the buffer immediately after use
    if((b = bread(dev, block + i)) != 0) {
      memmove(b->data, buf + (i * BSIZE), BSIZE);
      bwrite(b);  // Write directly to disk
      brelse(b);  // Release immediately
    } else {
      // cprintf("swapwrite: Could not allocate buffer for block %d\n", block + i);
      break;
    }
  }
}

// Similarly for swapread function

// Bypass log for swap operations - reading from disk
void
swapread(uint dev, char *buf, uint block, uint count)
{
  struct buf *b;
  
  if(buf == 0) {
    // cprintf("ERROR: swapread: Null buffer pointer\n");
    return;
  }
  
  for(int i = 0; i < count; i++){
    b = pageswap_bget(dev, block + i);
    if(b == 0) {
      // cprintf("ERROR: swapread: failed to get buffer for block %d\n", block + i);
      return;
    }
    memmove(buf + (i * BSIZE), b->data, BSIZE);
    brelse(b);
  }
}

// Find the victim process (highest RSS, or lowest PID if tied)
struct proc*
find_victim_proc(void)
{
  struct proc *p;
  struct proc *victim = 0;
  int max_rss = -1;

  acquire(&ptable.lock);
  // Loop through process table to find victim
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++){
    if(p->state == RUNNABLE || p->state == RUNNING || p->state == SLEEPING){
      if(p->pid >= 1 && (p->rss > max_rss || (p->rss == max_rss && victim && p->pid < victim->pid))){
        max_rss = p->rss;
        victim = p;
      }
    }
  }
  release(&ptable.lock);
  return victim;
}

// Find a victim page from the given process
// Returns: VA of the victim page, or 0 if none found
uint
find_victim_page(struct proc *p)
{
  pte_t *pte;
  uint va;
  if(p == 0) {
    // cprintf("ERROR: find_victim_page: Null process pointer\n");
    return 0;
  }
  // cprintf("find_victim_page: Scanning user space for victim page.\n");
  // Only scan user space (below KERNBASE)
  for(va = PGSIZE; va < KERNBASE; va += PGSIZE){
    pte = pageswap_walkpgdir(p->pgdir, (void*)va, 0);
    // Skip if page not present or not in user space
    if(!pte || !(*pte & PTE_P) || !(*pte & PTE_U))
      continue;
    // Look for pages with P bit set and A bit clear (not accessed recently)
    if((*pte & PTE_P) && !(*pte & PTE_A)){
      // cprintf("find_victim_page: Found victim page at va %x.\n", va);
      return va;
    }
  }
  // cprintf("find_victim_page: No pages with A bit clear. Clearing all A bits and retrying.\n");
  // If no page with A bit clear, clear all A bits and try again
  for(va = PGSIZE; va < KERNBASE; va += PGSIZE){
    pte = pageswap_walkpgdir(p->pgdir, (void*)va, 0);
    if(pte && (*pte & PTE_P) && (*pte & PTE_U)){
      *pte &= ~PTE_A;  // Clear access bit 
    }
  }
  // cprintf("find_victim_page: Retry scanning after clearing A bits.\n");

  // Return first page after clearing access bits (excluding VA 0)
  for(va = PGSIZE; va < KERNBASE; va += PGSIZE){ // Start from PGSIZE to avoid VA 0
    pte = pageswap_walkpgdir(p->pgdir, (void*)va, 0);
    if(pte && (*pte & PTE_P) && (*pte & PTE_U)){
      // cprintf("find_victim_page: Found victim page after clearing A bit at va %x.\n", va);
      return va;
    }
  }
  // cprintf("find_victim_page: No suitable victim page found.\n");
  return 0; // No suitable victim page found
}

// Count free memory pages
int 
count_free_pages(void)
{
  extern int get_free_page_count(void);
  int free_pages = get_free_page_count();
  // cprintf("count_free_pages: Free memory pages = %d\n", free_pages);
  return free_pages;
}

// Swap out a page to disk
// Returns 0 on success, -1 on failure
int
swapout(void)
{
  struct proc *victim_proc;
  uint victim_va;
  pte_t *pte;
  char *addr;
  int slotno;
  int perm;
  int npages_swapped = 0;
  
  // cprintf("\n=============== SWAP OUT OPERATION STARTED ===============\n");
  // print_memory_stats();
  
  // Find victim process
  victim_proc = find_victim_proc();
  if(!victim_proc) {
    // cprintf("ERROR: swapout: No victim process found\n");
    return -1;
  }

  // print_page_table(victim_proc);
  
  // cprintf("swapout: Victim process PID %d with RSS %d\n", victim_proc->pid, victim_proc->rss);
  
  // Try to swap out NPG pages
  while(npages_swapped < NPG){
    // Find victim page
    victim_va = find_victim_page(victim_proc);
    if(victim_va == 0) {
      break;
    }
    // Get PTE and permissions
    pte = pageswap_walkpgdir(victim_proc->pgdir, (void*)victim_va, 0);
    if(!pte || !(*pte & PTE_P)) {
      continue;
    }
    perm = PTE_FLAGS(*pte);

    // Skip kernel pages or other crucial pages
    if(!(perm & PTE_U)) {
      // cprintf("swapout: Skipping kernel page at 0x%x\n", victim_va);
      continue;
    }
    
    // Get physical address
    uint pa = PTE_ADDR(*pte);
    if(pa == 0) {
      continue;
    }
    addr = P2V(pa);
    // Swap out to disk
    slotno = swapout_page(addr, perm);
    if(slotno < 0) {
      continue;
    }
      
    // Update PTE to mark page as swapped
    // uint old_pte = *pte;
    *pte = PTE_SWAPPED | (slotno << 12) | (perm & ~PTE_P);
    
    // cprintf("swapout: Updated PTE from 0x%x to 0x%x for slot %d\n", old_pte, *pte, slotno);
    
    // Free physical memory
    kfree(addr);
    
    // Update RSS count
    victim_proc->rss--;
    
    npages_swapped++;
  }
  
  // If any pages were swapped, flush TLB
  if(npages_swapped > 0 && victim_proc->pid > 0) {
    // cprintf("swapout: Flushing TLB after swapping out %d pages\n", npages_swapped);
    lcr3(V2P(victim_proc->pgdir));
  }
  
  // cprintf("swapout: Completed swapping out %d pages\n", npages_swapped);
  // print_memory_stats();
  // cprintf(" =============== SWAP OUT OPERATION COMPLETED ===============\n");
  
  return (npages_swapped > 0) ? 0 : -1;
}


// Handle page fault for swapped page
// Returns 0 on success, -1 on failure
int
handle_pgfault(uint fault_addr)
{
  pte_t *pte;
  int slotno;
  char *mem;
  int perm;
  struct proc *curproc = myproc();
  if(curproc == 0) {
    return -1;
  }
  
  // cprintf("\n=== PAGE FAULT HANDLING STARTED ===\n");
  // cprintf("handle_pgfault: Address 0x%x, Process PID %d\n", fault_addr, curproc->pid);
  
  // Get PTE for faulting address
  uint page_addr = PGROUNDDOWN(fault_addr);
  pte = pageswap_walkpgdir(curproc->pgdir, (void*)page_addr, 0);
  if(!pte) {
    // cprintf("ERROR: handle_pgfault: No PTE found for address 0x%x\n", page_addr);
    return -1;
  }
  // Check if this is a swapped page
  if(!(*pte & PTE_SWAPPED)) {
    // cprintf("ERROR: handle_pgfault: Page at 0x%x is not swapped (PTE=0x%x)\n", page_addr, *pte);
    return -1;
  }
  // Extract slot number from PTE
  slotno = (*pte >> 12) & 0x3FF;  // Get bits 12-21 (10 bits) for slot number
  
  if(slotno < 0 || slotno >= NSWAPSLOTS) {
    // cprintf("ERROR: handle_pgfault: Invalid slot number %d from PTE 0x%x\n", slotno, *pte);
    return -1;
  }
  
  // cprintf("handle_pgfault: Page was swapped to slot %d\n", slotno);
  
  // Allocate new physical page
  mem = kalloc();
  if(!mem) {
    if(swapout() < 0 || (mem = kalloc()) == 0) {
      return -1;
    }
    // cprintf("ERROR: handle_pgfault: Failed to allocate physical page\n");
    // return -1;
  }
  
  // Read page from swap
  perm = swapin_page(mem, slotno);
  if(perm < 0) {
    // cprintf("ERROR: handle_pgfault: Failed to read page from swap slot %d\n", slotno);
    kfree(mem);
    return -1;
  }
  
  // Map the new page in page table with original permissions
  if(pageswap_mappages(curproc->pgdir, (void*)page_addr, PGSIZE, V2P(mem), perm | PTE_P) < 0) {
    // cprintf("ERROR: handle_pgfault: Failed to map page at address 0x%x\n", page_addr);
    kfree(mem);
    return -1;
  }
  
  // Get new PTE value for debugging
  pte = pageswap_walkpgdir(curproc->pgdir, (void*)page_addr, 0);
  
  // Clear the PTE_SWAPPED flag
  if(pte) {
    *pte &= ~PTE_SWAPPED;
  }
  
  // Update RSS count
  curproc->rss++;
  
  // Free the swap slot
  freeswapslot(slotno);
  
  // cprintf("handle_pgfault: Successfully handled page fault for address 0x%x\n", fault_addr);
  // print_memory_stats();
  // cprintf("=== PAGE FAULT HANDLING COMPLETED ===\n\n");
  
  return 0;
}


// Check if memory is low and swap out pages if needed
void
check_memory(void)
{
  int free_pages = count_free_pages();
  if(free_pages < TH){
    cprintf("Current Threshold = %d, Swapping %d pages\n", TH, NPG);
    swapout();
    
    TH = (TH * (100 - BETA)) / 100;
    int new_npg = (NPG * (100 + ALPHA)) / 100;
    if(new_npg > LIMIT)
      new_npg = LIMIT;
    NPG = new_npg;
    // cprintf("new TH = %d, new NPG = %d\n", TH, new_npg);

    
    // cprintf("check_memory: Threshold updated to %d, number of pages to swap out updated to %d.\n", TH, NPG);
  }
}

// Free all swap slots used by a process that's exiting
void
free_process_swap_slots(void)
{
  pte_t *pte;
  uint va;
  struct proc *curproc = myproc();

  if(curproc == 0) { return;}

  // Look through the page table for swapped pages
  for(va = 0; va < KERNBASE; va += PGSIZE){
    pte = pageswap_walkpgdir(curproc->pgdir, (void*)va, 0);
    
    if(pte && (*pte & PTE_SWAPPED)){
      int slotno = (*pte >> 12) & 0x3FF;  // Extract slot number

      if(slotno >= 0 && slotno < NSWAPSLOTS) {
        freeswapslot(slotno);
        *pte = 0;  // Clear the PTE
      }
      
      // cprintf("free_process_swap_slots: Freed swap slot %d for page at va %x.\n", slotno, va);
    }
  }
}

void
handle_trap_pgfault(struct trapframe *tf)
{
  uint fault_addr = rcr2();
  // Check if this is a user space page fault
  if((tf->cs & 3) == DPL_USER) {
    // Handle page fault for user space
    if(handle_pgfault(fault_addr) < 0) {
      // If we couldn't handle the page fault, kill the process
      // cprintf("pid %d %s: trap %d err %d on cpu %d ""eip 0x%x addr 0x%x--kill proc\n",myproc()->pid, myproc()->name, tf->trapno,tf->err, cpuid(), tf->eip, fault_addr);
      myproc()->killed = 1;
    }
  } else {
    // cprintf("kernel page fault at %p\n", fault_addr);
    panic("kernel page fault");
  }
}




//Debuging
void print_memory_stats(void) {
  int free_pages = count_free_pages();
  struct proc *p;
  int total_rss = 0;
  int swapped_pages = 0;
  int proc_count = 0;
  int running_count = 0;
  
  // Count used swap slots
  acquire(&swaplock);
  for(int i = 0; i < NSWAPSLOTS; i++) {
    if(!swapslots[i].is_free) {
      swapped_pages++;
    }
  }
  release(&swaplock);
  
  // Count total RSS and processes
  acquire(&ptable.lock);
  for(p = ptable.proc; p < &ptable.proc[NPROC]; p++) {
    if(p->state != UNUSED) {
      proc_count++;
      total_rss += p->rss;
      if(p->state == RUNNING || p->state == RUNNABLE) {
        running_count++;
      }
    }
  }
  release(&ptable.lock);
  
  cprintf("--- MEMORY STATS ---\n");
  cprintf("Free pages:        %d\n", free_pages);
  cprintf("Total RSS:         %d\n", total_rss);
  cprintf("Swap slots used:   %d/%d\n", swapped_pages, NSWAPSLOTS);
  cprintf("Active processes:  %d (running/runnable: %d)\n", proc_count, running_count);
  cprintf("Current TH/NPG:    %d/%d\n", TH, NPG);
  cprintf("-------------------\n");
  
  // If swap slots are almost full, print detailed slot info
  if(swapped_pages > (NSWAPSLOTS - 20)) {
    cprintf("SWAP SLOTS NEARLY FULL - Displaying detail:\n");
    acquire(&swaplock);
    int free_count = 0;
    for(int i = 0; i < NSWAPSLOTS; i++) {
      if(swapslots[i].is_free) free_count++;
    }
    cprintf("First 10 slots status: ");
    for(int i = 0; i < 10; i++) {
      cprintf("[%d:%d] ", i, swapslots[i].is_free);
    }
    cprintf("\nLast 10 slots status: ");
    for(int i = NSWAPSLOTS-10; i < NSWAPSLOTS; i++) {
      cprintf("[%d:%d] ", i, swapslots[i].is_free);
    }
    cprintf("\nTotal free slots: %d\n", free_count);
    release(&swaplock);
  }
}


void print_page_table(struct proc *p) {
  if (!p) {
    cprintf("Error: No process specified.\n");
    return;
  }
  
  cprintf("Page Table for Process PID %d:\n", p->pid);
  cprintf("Virtual Address\t\tPhysical Address\t\tFlags\n");
  cprintf("----------------------------------------------------------\n");

  for (int i = 0; i < NPG; i++) { // NPG: number of pages for a process
    pte_t *pte = pageswap_walkpgdir(p->pgdir, (void*)(i * PGSIZE), 0);
    if (!pte) {
      continue; // No PTE available for this virtual address.
    }
    
    if (*pte & PTE_P) { // Check if the page is present
      uint pa = PTE_ADDR(*pte); // Extract the physical address from the PTE
      uint flags = *pte & 0xFFF; // Get flags 
      cprintf("0x%x\t\t0x%x\t\t0x%x\n", i * PGSIZE, pa, flags);
    }
  }
}
// Per-CPU state
struct cpu {
  uchar apicid;                // Local APIC ID
  struct context *scheduler;   // swtch() here to enter scheduler
  struct taskstate ts;         // Used by x86 to find stack for interrupt
  struct segdesc gdt[NSEGS];   // x86 global descriptor table
  volatile uint started;       // Has the CPU started?
  int ncli;                    // Depth of pushcli nesting.
  int intena;                  // Were interrupts enabled before pushcli?
  struct proc *proc;           // The process running on this cpu or null
};

extern struct cpu cpus[NCPU];
extern int ncpu;

//PAGEBREAK: 17
// Saved registers for kernel context switches.
// Don't need to save all the segment registers (%cs, etc),
// because they are constant across kernel contexts.
// Don't need to save %eax, %ecx, %edx, because the
// x86 convention is that the caller has saved them.
// Contexts are stored at the bottom of the stack they
// describe; the stack pointer is the address of the context.
// The layout of the context matches the layout of the stack in swtch.S
// at the "Switch stacks" comment. Switch doesn't save eip explicitly,
// but it is on the stack and allocproc() manipulates it.
struct context {
  uint edi;
  uint esi;
  uint ebx;
  uint ebp;
  uint eip;
};

enum procstate { UNUSED, EMBRYO, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

typedef void (*sighandler_t)(void);

// Per-process state
struct proc {
  uint sz;                     // Size of process memory (bytes)
  pde_t* pgdir;                // Page table
  char *kstack;                // Bottom of kernel stack for this process
  enum procstate state;        // Process state
  int pid;                     // Process ID
  struct proc *parent;         // Parent process
  struct trapframe *tf;        // Trap frame for current syscall
  struct trapframe *saved_tf;  // For signal handling
  struct context *context;     // swtch() here to run process
  void *chan;                  // If non-zero, sleeping on chan
  int killed;                  // If non-zero, have been killed
  struct file *ofile[NOFILE];  // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)
  
  int suspended;               
  sighandler_t sighandler; 
  int pending_custom_signal;
  uint signal_frame_eip;

  int start_later;         // start later for custom_fork
  int exec_time;           // Execution time in ticks (-1 for indefinite)
  int started;       

  // Priority scheduling fields
  int init_priority;    // Initial priority πi(0)
  int dynamic_priority; // Current dynamic priority πi(t)
  int wait_ticks;       // Wait time Wi(t)
  int cpu_ticks;        // CPU ticks consumed Ci(t)
  
  //scheduler profile
  int creation_time;
  int start_time;
  int end_time;
  int context_switches;
  int has_started;
  int response_time;
  int last_runnable;
};

// Process memory is laid out contiguously, low addresses first:
//   text
//   original data and bss
//   fixed-size stack
//   expandable heap

void update_wait_times(void);

enum interrupt { SIGINT, SIGBG, SIGFG, SIGCUSTOM };
int keyinterrupt(enum interrupt signal);
void suspend(void);
int custom_fork(int start_later, int exec_time);
int scheduler_start(void);

void update_priorities(void);
struct proc* find_highest_priority_process(void);
void update_wait_ticks(void);
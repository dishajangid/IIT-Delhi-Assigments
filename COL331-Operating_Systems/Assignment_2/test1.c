#include "types.h"
#include "stat.h"
#include "user.h"

int fib(int n) {
  if(n <= 0) return 0;
  if(n == 1) return 1;
  if(n == 2) return 1;
  return fib(n-1) + fib(n-2);
}

void sv() {
  printf(1, "I am Shivam\n");
}

void myHandler() {
  printf(1, "I am inside the handler\n");
  sv();
}

int main() {
    int i = 0;
    signal(myHandler); // register handler
    while(1) {
        printf(1, "%d This is normal code running\n", i);
        for(int j = 0; j < 100000; j++){}
        i++;
        fib(5); // CPU intensive work
    }
}

#include "types.h"
#include "stat.h"
#include "user.h"

int main(void) {
  int i = 0;
  while(1) {
    printf(1, "%d Running... \n", i);
    i++;
    sleep(100);
  }
}

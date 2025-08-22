// init: The initial user-level program

#include "types.h"
#include "stat.h"
#include "user.h"
#include "fcntl.h"

char *argv[] = { "sh", 0 };

#define MAX_ATTEMPTS 3

int login() {
    char input_username[64];
    char input_password[64];
    int attempts = 0;

    printf(1, "Enter username: ");
    gets(input_username, sizeof(input_username));
    input_username[strlen(input_username) - 1] = '\0';  

    while (strcmp(input_username, USERNAME) != 0) {
        printf(1, "Invalid username. Try again.\n");
        printf(1, "Enter username: ");
        gets(input_username, sizeof(input_username));
        input_username[strlen(input_username) - 1] = '\0'; 
    }

    while (attempts < MAX_ATTEMPTS) {
        printf(1, "Enter password: ");
        gets(input_password, sizeof(input_password));
        input_password[strlen(input_password) - 1] = '\0'; 

        if (strcmp(input_password, PASSWORD) == 0) {
            printf(1, "Login successful\n");
            return 1; 
        } else {
            attempts++;
            printf(1, "Incorrect password. Attempt %d of %d\n", attempts, MAX_ATTEMPTS);
        }
    }
    printf(1, "Login failed. System locked.\n");
    return 0; 
}


int main(void) {
  int pid, wpid;

  if(open("console", O_RDWR) < 0){
    mknod("console", 1, 1);
    open("console", O_RDWR);
  }
  dup(0);  // stdout
  dup(0);  // stderr

  if (login() == 0) {
      exit();
  }

  for(;;){
    printf(1, "init: starting sh\n");
    pid = fork();
    if(pid < 0){
      printf(1, "init: fork failed\n");
      exit();
    }
    if(pid == 0){
      exec("sh", argv);
      printf(1, "init: exec sh failed\n");
      exit();
    }
    while((wpid=wait()) >= 0 && wpid != pid)
      printf(1, "zombie!\n");
  }
}

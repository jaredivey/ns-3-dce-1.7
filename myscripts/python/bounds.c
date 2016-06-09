#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/resource.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#define PAGESIZE 4096

extern end;
extern etext;
extern edata;

static jmp_buf back;

void sighandler(int sig) {
	longjmp(back, 1);
}

void find_limit(const void *start, int up) {
  char work[4096];
  FILE* f;
  char* pCh;
  int i = 0;

  sprintf(work, "/proc/self/stat");
  f = fopen(work, "r");
  if (f == NULL)
    {
      printf("Can't open %s\n", work);
      //return(0);
    }
  if(fgets(work, sizeof(work), f) == NULL)
    printf("Error with fgets\n");
  fclose(f);
  strtok(work, " ");
  for (i = 1; i < 45; i++)
    {
      pCh = strtok(NULL, " ");
      printf ("%s\n", pCh);
    }

  printf("Start: %s\n", pCh);
  printf("End:   %s\n", strtok(NULL, " "));
}

int main(int argc, char **argv) {
	find_limit(&argc, 0);
	//find_limit(&argc, 1);
	return 0;
}

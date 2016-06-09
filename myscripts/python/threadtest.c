#include <sys/types.h>   /* for type definitions */
#include <sys/socket.h>  /* for socket API function calls */
#include <netinet/in.h>  /* for address structs */
#include <arpa/inet.h>   /* for sockaddr_in */
#include <stdio.h>       /* for printf() */
#include <stdlib.h>      /* for atoi() */
#include <string.h>      /* for strlen() */
#include <unistd.h>      /* for close() */
#include <netdb.h>
#include <errno.h>
#include <pthread.h>

#define MAXBUFFLEN 1000

struct stuff {
   int s;
   int n;
   int m;
};

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int client (char *address, int port, int buffer_size)
{
  int sock;
  sock = socket (PF_INET, SOCK_STREAM, 0);
  int total = 0;
  int result;
  struct hostent *host = gethostbyname (address);

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons (port);

  memcpy (&addr.sin_addr.s_addr, host->h_addr_list[0], host->h_length);

  result = connect (sock, (const struct sockaddr *) &addr, sizeof (addr));

  uint8_t buffer[MAXBUFFLEN];

  memset (buffer, 'a', MAXBUFFLEN-1);
  ssize_t tot = 0;

  total = write (sock, buffer, buffer_size);
  printf("write: %d\n", total);

  close (sock);

  return total;
}

/* this function is run by the second thread */
void *print_it(void *ptr)
{
    struct stuff *stuff_ptr = (struct stuff *)ptr;
    int i = 0;
    for (i = stuff_ptr->s; i < stuff_ptr->m; i+=stuff_ptr->n)
    {
        printf("[ID %d] %d\n", stuff_ptr->s, i);
    }

    /* the function must return something - NULL will do */
    return NULL;
}

int main(int argc, char *argv[])
{
    int portno;
    int total = 0;

    struct stuff tt;
    int s = 0;
    int n = 500;
    int m = 500000;

    /* this variable is our reference to the second thread */
    pthread_t threads[500000];

    for (s = 0; s < n; s++)
    {
        tt.s = s; tt.n = n; tt.m = m;
        if(pthread_create(&threads[s], NULL, print_it, &tt)) {
            fprintf(stderr, "Error creating thread\n");
            return 1;
        }
    }

    /* wait for the threads to finish */
    for (s = 0; s < n; s++)
    {
        if(pthread_join(threads[s], NULL)) {
            fprintf(stderr, "Error joining thread\n");
            return 2;
        }
    }

    if (argc > 1)
    {
        portno = atoi(argv[1]);
        total += client ("127.0.0.1", portno, MAXBUFFLEN);
        printf ("Sent %d bytes\n", total);
    }

    return 0;

}

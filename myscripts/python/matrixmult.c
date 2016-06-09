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

#define MAXBUFFLEN 1000

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

int main(int argc, char *argv[])
{
    int matrix_prod[MAXBUFFLEN][MAXBUFFLEN] = {0};
    int matrix_array[MAXBUFFLEN][MAXBUFFLEN] = {0};

    int i, j, k;
    int total = 0;
    int portno;

    // Load the vector with consecutive values.
    for (i = 0; i < MAXBUFFLEN; ++i)
    {
        for (j = 0; j < MAXBUFFLEN; ++j)
        {
            matrix_array[i][j] = i*MAXBUFFLEN + j;
        }
    }

    for (i = 0; i < MAXBUFFLEN; ++i)
    {
        for (j = 0; j < MAXBUFFLEN; ++j)
        {
            for (k = 0; k < MAXBUFFLEN; ++k)
            {
                matrix_prod[i][j] += matrix_array[i][k]*matrix_array[k][j];
            }            
        }
    }

    for (i = 0; i < MAXBUFFLEN; ++i)
    {
        for (j = 0; j < MAXBUFFLEN; ++j)
        {
            printf ("%d\t", matrix_prod[i][j]);           
        }
        printf("\n");
    }

    if (argc > 1)
    {
        portno = atoi(argv[1]);
        total += client ("127.0.0.1", portno, MAXBUFFLEN);
        printf ("Sent %d bytes\n", total);
    }
}

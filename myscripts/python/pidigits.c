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
#define DIGITS 25000
#define SCALE 10000
#define ARRINIT 2000 

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
    int i, j, k;
    int total = 0;
    int portno;

    int arr[DIGITS + 1];
    int carry = 0;
    int sum = 0;

    for (i = 0; i <= DIGITS; ++i)
        arr[i] = ARRINIT;

    for (i = DIGITS; i > 0; i-=14)
    {
        sum = 0;
        for (j = i; j > 0; --j)
        {
            sum = sum *j + SCALE * arr[j];
            arr[j] = sum % (j * 2 - 1);
            sum /= j * 2 - 1;
        }
        printf ("%04d", carry + sum / SCALE);
        carry = sum % SCALE;
    }


    if (argc > 1)
    {
        portno = atoi(argv[1]);
        total += client ("127.0.0.1", portno, MAXBUFFLEN);
        printf ("Sent %d bytes\n", total);
    }
}

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <errno.h>

#define MAXBUFFLEN 16777216

int main(int argc, char *argv[])
{
    int sockfd, portno, n, buffer_size, loop_count, i;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    int total = 0;

    char *buffer = NULL;
    if (argc < 5) {
       fprintf(stderr,"usage %s hostname port bufsize\n", argv[0]);
       exit(0);
    }
    portno = atoi(argv[2]);
    buffer_size = atoi(argv[3]);
    loop_count = atoi(argv[4]);
    if (buffer_size > MAXBUFFLEN)
    {
       fprintf(stderr,"buffer too large\n");
       exit(0);
    }
    buffer = (char *)calloc (sizeof(char), buffer_size);
    memset (buffer, 'a', buffer_size);

    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    memset((char *) &serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(portno);
    memcpy (&serv_addr.sin_addr.s_addr, server->h_addr_list[0], server->h_length);

    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        fprintf(stderr,"ERROR opening socket ? errno %d\n", errno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        fprintf(stderr, "ERROR connecting ? errno %d\n", errno);
    for (i = 0; i < loop_count; ++i)
    {
        n = write(sockfd,buffer,buffer_size);
        if (n < 0) 
             fprintf(stderr, "ERROR writing to socket ? errno %d\n", errno);
        printf ("Sent %d bytes\n", n);
        total += n;
    }
    close(sockfd);
    printf ("Sent %d bytes\n", total);
}

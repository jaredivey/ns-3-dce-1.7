#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <errno.h>

#define MAXBUFFLEN 16777216

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

void client_side (int argc, char *argv[], char *buffer, int *write_size)
{
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    if (argc < 3) {
       fprintf(stderr,"usage %s hostname port\n", argv[0]);
       exit(0);
    }
    portno = atoi(argv[2])+1;
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    memset((char *) &serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy (&serv_addr.sin_addr.s_addr, server->h_addr_list[0], server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0) 
        error("ERROR connecting");

    do
    {
        memset (buffer, 'a', *write_size);
        n = write(sockfd,buffer,*write_size);
        if (n < 0) 
            error("ERROR writing to socket");
        *write_size -= n;
    } while (*write_size > 0);

    (*write_size) = n;
    close(sockfd);
}

void server_side (int argc, char *argv[], char *buffer, int *read_size)
{
    int sockfd, newsockfd, portno;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    int total_sent = 0;
    int total_recv = 0;
    int cur_recv = 0;
    int max_recv = 0;
    if (argc < 4) {
        fprintf(stderr,"usage %s hostname port max_recv\n", argv[0]);
        exit(1);
    }
    max_recv = atoi(argv[3]);
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    memset((char *) &serv_addr, 0, sizeof(serv_addr));
    portno = atoi(argv[2]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) 
        error("ERROR on binding");
    listen(sockfd,256);
    clilen = sizeof(cli_addr);

    buffer = (char *)calloc (sizeof(char), MAXBUFFLEN);
    while (1)
    {
        newsockfd = accept(sockfd, 0, 0);
        if (newsockfd < 0) 
            error("ERROR on accept");
        memset(buffer,0,MAXBUFFLEN);
        cur_recv = 0;
        do
        {
            n = read(newsockfd,buffer,MAXBUFFLEN);
            if (n < 0) error("ERROR reading from socket");

            cur_recv += n;
            printf ("Received %d bytes\n", cur_recv);
        } while (n > 0);
        close(newsockfd);
        printf ("Received %d bytes\n", total_recv);
        total_recv += cur_recv;
        total_sent += cur_recv;
        printf ("Sent %d bytes\n", total_sent);

        (*read_size) = n;

        if (total_recv >= max_recv)
        {
            printf ("Time to send, received %d bytes\n", total_recv);
            break;
        }
     }
     close(sockfd);

     client_side (argc, argv, buffer, &total_sent);
}

int main(int argc, char *argv[])
{
     char *buffer = NULL;
     int buffer_size = 0;

     server_side (argc, argv, buffer, &buffer_size);
}

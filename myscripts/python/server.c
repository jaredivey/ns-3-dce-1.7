#include <sys/types.h>  /* for type definitions */
#include <sys/socket.h> /* for socket API calls */
#include <netinet/in.h> /* for address structs */
#include <arpa/inet.h>  /* for sockaddr_in */
#include <stdio.h>      /* for printf() and fprintf() */
#include <stdlib.h>     /* for atoi() */
#include <string.h>     /* for strlen() */
#include <unistd.h>     /* for close() */
#include <netdb.h>

#define MAXBUFFLEN 16384

int main(int argc, char *argv[])
{
    int sockfd;
    int newsockfd;
    int portno;
    socklen_t clilen;
    char buffer[MAXBUFFLEN];
    struct sockaddr_in serv_addr;
    struct sockaddr_in cli_addr;
    int n = MAXBUFFLEN;
    int total = 0;
    if (argc < 2) {
        fprintf(stderr,"ERROR, no port provided\n");
        exit(1);
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
    {
        perror("ERROR opening socket");
        exit(1);
    }
    memset((char *) &serv_addr, 0, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
            sizeof(serv_addr)) < 0)
    {
        perror("ERROR on binding");
        exit(1);
    }
    listen(sockfd,5);
    clilen = sizeof(cli_addr);

    while (1)
    {
        newsockfd = accept(sockfd, 
                (struct sockaddr *) &cli_addr, 
                &clilen);
        if (newsockfd < 0)
        {
            perror("ERROR on accept");
            exit(1);
        }
        memset(buffer,0, n);
        n = read(newsockfd,buffer,MAXBUFFLEN);
        if (n < 0)
        {
            perror("ERROR reading from socket");
            exit(1);
        }

        total += n;
        printf("Received %d bytes\n",total);
        close(newsockfd);
    }
    close(sockfd);
}

#include <sys/types.h>   /* for type definitions */
#include <sys/socket.h>  /* for socket API function calls */
#include <netinet/in.h>  /* for address structs */
#include <arpa/inet.h>   /* for sockaddr_in */
#include <stdio.h>       /* for printf() */
#include <stdlib.h>      /* for atoi() */
#include <string.h>      /* for strlen() */
#include <unistd.h>      /* for close() */
#include <netdb.h>
#include <math.h>
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
    int i, j, k;
    int total = 0;
    int portno;

      /* screen ( integer) coordinate */
    int iX,iY;
    const int iXmax = 10000; 
    const int iYmax = 10000;
    /* world ( double) coordinate = parameter plane*/
    double Cx,Cy;
    const double CxMin=-2.5;
    const double CxMax=1.5;
    const double CyMin=-2.0;
    const double CyMax=2.0;
    /* */
    double PixelWidth=(CxMax-CxMin)/iXmax;
    double PixelHeight=(CyMax-CyMin)/iYmax;
    /* color component ( R or G or B) is coded from 0 to 255 */
    /* it is 24 bit color RGB file */
    const int MaxColorComponentValue=255; 
    FILE * fp;
    char *filename="mandelbrot_c.ppm";
    char *comment="# ";/* comment should start with # */
    static unsigned char color[3];
    /* Z=Zx+Zy*i  ;   Z0 = 0 */
    double Zx, Zy;
    double Zx2, Zy2; /* Zx2=Zx*Zx;  Zy2=Zy*Zy  */
    /*  */
    int Iteration;
    const int IterationMax=200;
    /* bail-out value , radius of circle ;  */
    const double EscapeRadius=2;
    double ER2=EscapeRadius*EscapeRadius;
    /*create new file,give it a name and open it in binary mode  */
    fp= fopen(filename,"wb"); /* b -  binary mode */
    /*write ASCII header to the file*/
    fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,iXmax,iYmax,MaxColorComponentValue);
    /* compute and write image data bytes to the file*/
    for(iY=0;iY<iYmax;iY++)
    {
         Cy=CyMin + iY*PixelHeight;
         if (fabs(Cy)< PixelHeight/2) Cy=0.0; /* Main antenna */
         for(iX=0;iX<iXmax;iX++)
         {         
                    Cx=CxMin + iX*PixelWidth;
                    /* initial value of orbit = critical point Z= 0 */
                    Zx=0.0;
                    Zy=0.0;
                    Zx2=Zx*Zx;
                    Zy2=Zy*Zy;
                    /* */
                    for (Iteration=0;Iteration<IterationMax && ((Zx2+Zy2)<ER2);Iteration++)
                    {
                        Zy=2*Zx*Zy + Cy;
                        Zx=Zx2-Zy2 +Cx;
                        Zx2=Zx*Zx;
                        Zy2=Zy*Zy;
                    };
                    /* compute  pixel color (24 bit = 3 bytes) */
                    if (Iteration==IterationMax)
                    { /*  interior of Mandelbrot set = black */
                       color[0]=0;
                       color[1]=0;
                       color[2]=0;                           
                    }
                 else 
                    { /* exterior of Mandelbrot set = white */
                         color[0]=255; /* Red*/
                         color[1]=255;  /* Green */ 
                         color[2]=255;/* Blue */
                    };
                    /*write color to the file*/
                    fwrite(color,1,3,fp);
            }
    }
    fclose(fp);


    if (argc > 1)
    {
        portno = atoi(argv[1]);
        total += client ("127.0.0.1", portno, MAXBUFFLEN);
        printf ("Sent %d bytes\n", total);
    }
}

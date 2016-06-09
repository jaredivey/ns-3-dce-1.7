import java.io.*;
import java.net.*;

class TCPServer {
    public static void main(String argv[]) throws IOException {
        if (argv.length > 0) {
	        int characters = 0;
	        int portno;
	        int total = 0;
	        try {
	            portno = Integer.parseInt(argv[0]);
	            ServerSocket welcomeSocket = new ServerSocket (portno);
	
				while (true) {
					try {
						characters = 0;
						Socket connectionSocket = welcomeSocket.accept();
						BufferedReader inFromClient = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));
						while (inFromClient.read() != -1) {
							characters++;
						}
						total += characters;
						System.out.println("Received: " + total + " bytes\n");
					}
					catch(SocketTimeoutException s) {
						System.out.println("Socket timed out!");
						break;
					}
					catch(IOException e) {
						e.printStackTrace();
						break;
					}
				}
				welcomeSocket.close();
	        } catch (NumberFormatException e) {
	            System.err.println("Argument " + argv[0] + " must be an integer.");
	            System.exit(1);
	        }
		}
	}
}

import java.io.*;
import java.net.*;

class TCPClientServer {
    public static void main(String argv[]) throws IOException {
        if (argv.length > 2) {
	    int portno;
            int max_recv;
	    int total_sent = 0;
	    int total_recv = 0;
            String ipAddress;
	    try {
                portno = Integer.parseInt(argv[1]);
                max_recv = Integer.parseInt(argv[2]);
                ServerSocket welcomeSocket = new ServerSocket (portno);
                while (true) {
                    try {
                        Socket connectionSocket = welcomeSocket.accept();
                        BufferedReader inFromClient = new BufferedReader(new InputStreamReader(connectionSocket.getInputStream()));
                        char[] recv_buf = new char[max_recv];
                        int cur_recv = inFromClient.read(recv_buf, 0, max_recv);
                        while (cur_recv != -1) {
                            total_recv += cur_recv;
                            cur_recv = inFromClient.read(recv_buf, 0, max_recv);
                        }
                        System.out.println("Received: " + total_recv + " bytes\n");

                        if (total_recv >= max_recv) {
                            break;
                        }
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

                byte[] buf = new byte[65536];
                ipAddress = argv[0];
                Socket clientSocket = new Socket(ipAddress, portno+1);
                DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
                while (total_recv > 0) {
                    outToServer.write(buf, 0, 65536);
                    total_recv -= 65536;
                }
                total_sent += total_recv;
                System.out.println("Sent " + total_sent + " bytes\n");
                clientSocket.close ();

                welcomeSocket.close ();
            } catch (NumberFormatException e) {
                System.err.println("Argument " + argv[1] + " must be an integer.");
                System.exit(1);
            }
        }
    }
}

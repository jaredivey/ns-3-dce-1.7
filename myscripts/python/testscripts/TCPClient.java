import java.io.*;
import java.net.*;

class TCPClient {
    public static void main(String argv[]) throws Exception {
        if (argv.length > 3) {
                int portno;
                int loopCount;
                int buffer_size;
                int total = 0;
        	String ipAddress;
                try {
                        portno = Integer.parseInt(argv[1]);
                        buffer_size = Integer.parseInt(argv[2]);
                        loopCount = Integer.parseInt(argv[3]);
                        ipAddress = argv[0];
                        byte[] buf = new byte[buffer_size];
                        Socket clientSocket = new Socket(ipAddress, portno);
                        DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
                        for (int i = 0; i < loopCount; ++i) {
                                outToServer.write(buf);
                                System.out.println("Sent " + buffer_size + " bytes\n");
                                total += buffer_size;
                        }
                        clientSocket.close ();
                        System.out.println("Sent " + total + " bytes\n");
                } catch (NumberFormatException e) {
                    System.err.println("Argument " + argv[1] + " and " + argv[2] + " must be an integer.");
                    System.exit(1);
                }
        }
    }
}

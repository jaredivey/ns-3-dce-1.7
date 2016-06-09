import java.io.*;
import java.net.*;

public class ThreadTest
{
    public static void main(String argv[]) throws Exception
    {
        int portno;
        int buffer_size = 1000;
        int total = 0;

        for (int i = 0; i < 500; ++i) {
            MyThread t1 = new MyThread(i, 1000, 500000);
            t1.start();
        }

        if (argv.length > 0) {
            try {
                portno = Integer.parseInt(argv[0]);
                Socket clientSocket = new Socket("localhost", portno);
                byte[] buf = new byte[buffer_size];
                DataOutputStream outToServer = new DataOutputStream(clientSocket.getOutputStream());
                outToServer.write(buf);
                System.out.println("Sent " + buffer_size + " bytes\n");
                total += buffer_size;
                clientSocket.close ();
                System.out.println("Sent " + total + " bytes\n");
            } catch (NumberFormatException e) {
                System.err.println("Argument " + argv[1] + " must be an integer.");
                System.exit(1);
            }
        }
    }
}

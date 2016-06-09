import java.io.*;
import java.net.*;

class PiDigits {
    public static void main(String argv[]) throws Exception {
        int i, j;
        int portno;
        int digits = 25000;
        int buffer_size = 1000;
        int total = 0;

        int SCALE = 10000;
        int ARRINIT = 2000;
        int[] arr = new int[digits+1];
        int carry = 0;

        for (i = 0; i <= digits; ++i) {
            arr[i] = ARRINIT;
        }

        for (i = digits; i > 0; i-=14) {
            int sum = 0;
            for (j = i; j > 0; --j) {
                sum = sum * j + SCALE * arr[j];
                arr[j] = sum % (j * 2 - 1);
                sum /= j * 2 - 1;
            }
            System.out.print (String.format("%04d", carry + sum / SCALE));
            carry = sum % SCALE;
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

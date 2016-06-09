import java.io.*;
import java.net.*;

class MatrixMult {
    public static void main(String argv[]) throws Exception {
        int i, j, k;
        int portno;
        int buffer_size = 1000;
        int total = 0;

        int[][] matrix_array = new int[buffer_size][buffer_size];
        int[][] matrix_prod = new int[buffer_size][buffer_size];

        for (i = 0; i < buffer_size; ++i) {
                for (j = 0; j < buffer_size; ++j) {
                        matrix_array[i][j] = i*buffer_size + j;
                        matrix_prod[i][j] = 0;
                }
        }

        for (i = 0; i < buffer_size; ++i) {
                for (j = 0; j < buffer_size; ++j) {
                        for (k = 0; k < buffer_size; ++k) {
                                matrix_prod[i][j] += matrix_array[i][k]*matrix_array[k][j];
                        }
                }
        }

        for (i = 0; i < buffer_size; ++i) {
                for (j = 0; j < buffer_size; ++j) {
                        System.out.print(matrix_prod[i][j]+"\t");
                }
                System.out.print("\n");
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

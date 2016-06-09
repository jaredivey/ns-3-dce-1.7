import java.io.*;
import java.net.*;

public class Ping {
    public static void main(String argv[]) throws UnknownHostException, IOException {
        if (argv.length > 2) {
        	try {
        String ipAddress = argv[0];
        int size = Integer.parseInt(argv[1]); // Unused for Java
        int count = Integer.parseInt(argv[2]);
        int i;
        int portno;
        int buffer_size = 0; // Use the buffer size to confirm ping successes
        InetAddress inet = InetAddress.getByName (ipAddress);

        System.out.println("Sending " + count + " Ping request(s) to " + ipAddress);
        for (i = 0; i < count; ++i) {
            if (inet.isReachable(1000)) {
                System.out.println(buffer_size + " Host is reachable");
                ++buffer_size;
            }
        }
            } catch (NumberFormatException e) {
                System.err.println("Argument " + argv[1] + " must be an integer.");
                System.exit(1);
            }
        }
    }
}

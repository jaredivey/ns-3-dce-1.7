import java.io.*;
import java.net.*;

public class TestPing {
    public static void main(String argv[]) throws UnknownHostException, IOException {
        if (argv.length > 2) {
        	try {
                String ipAddress = argv[0];
                int size = Integer.parseInt(argv[1]); // Unused for Java
                int count = Integer.parseInt(argv[2]);
                int i;
                InetAddress inet = InetAddress.getByName (ipAddress);

                System.out.println("Sending " + count + " Ping request(s) to " + ipAddress);
                for (i = 0; i < count; ++i) {
                    if (inet.isReachable(1000)) {
                        System.out.println(i + " Host is reachable");
                    }
                 }
            } catch (NumberFormatException e) {
                System.err.println("Arguments " + argv[1] + " and " + argv[2] + " must be integers.");
                System.exit(1);
            }
        }
    }
}

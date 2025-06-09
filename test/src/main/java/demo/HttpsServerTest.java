package demo;

import com.sun.net.httpserver.HttpServer;
import lombok.val;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.security.KeyManagementException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableKeyException;
import java.security.cert.CertificateException;

/**
 * @author bin
 * @since 2025/06/09
 */
public class HttpsServerTest {
    public static void main(String[] args)
            throws IOException, KeyStoreException, NoSuchAlgorithmException, UnrecoverableKeyException,
            CertificateException, KeyManagementException {
        val server = HttpServer.create(new InetSocketAddress(8080), 0);
        server.createContext("/", exchange -> {
            String response = "Hello, HTTPS World!";
            exchange.sendResponseHeaders(200, response.length());
            exchange.getResponseBody().write(response.getBytes());
            exchange.close();
        });
        server.start();
    }
}

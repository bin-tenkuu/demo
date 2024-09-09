package demo.websocket;

import lombok.val;
import org.java_websocket.WebSocket;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.handshake.ServerHandshake;
import org.java_websocket.server.WebSocketServer;

import java.net.InetSocketAddress;
import java.net.URI;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/09
 */
public class WebSocketTest {
    public static void main(String[] args) throws InterruptedException {
        val server = new Server(8880);
        server.setReuseAddr(true);
        server.start();

        val client = new Client("ws://127.0.0.1:8880");
        Thread.sleep(1000);
        client.connect();
        Thread.sleep(1000);
        client.sendPing();
        Thread.sleep(1000);
        client.send("hello");
        Thread.sleep(1000);
        client.close();

        Thread.sleep(1000);
        server.stop();
    }

    private static final class Server extends WebSocketServer {
        public Server(int port) {
            super(new InetSocketAddress(port));
        }

        @Override
        public void onOpen(WebSocket webSocket, ClientHandshake clientHandshake) {
            System.out.println("open");
        }

        @Override
        public void onClose(WebSocket webSocket, int i, String s, boolean b) {
            System.out.println("close");
        }

        @Override
        public void onMessage(WebSocket webSocket, String s) {
            System.out.println(s);
        }

        @Override
        public void onError(WebSocket webSocket, Exception e) {
            System.out.println("error");
            e.printStackTrace();
        }

        @Override
        public void onStart() {
            System.out.println("start");
        }
    }

    private static final class Client extends WebSocketClient {
        public Client(String uri) {
            super(URI.create(uri));
        }

        @Override
        public void onOpen(ServerHandshake handshakedata) {
            System.out.println("open");
        }

        @Override
        public void onMessage(String message) {
            System.out.println(message);
        }

        @Override
        public void onClose(int code, String reason, boolean remote) {
            System.out.println("close");
        }

        @Override
        public void onError(Exception e) {
            System.out.println("error");
            e.printStackTrace();
        }
    }
}

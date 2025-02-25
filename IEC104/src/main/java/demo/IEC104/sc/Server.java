package demo.IEC104.sc;

import demo.IEC104.ByteUtil;
import demo.IEC104.FrameUtil;
import lombok.val;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.channels.AsynchronousServerSocketChannel;
import java.nio.channels.AsynchronousSocketChannel;
import java.nio.channels.CompletionHandler;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@SuppressWarnings("CallToPrintStackTrace")
public class Server implements CompletionHandler<AsynchronousSocketChannel, Void>, Closeable {
    private final AsynchronousServerSocketChannel serverSocketChannel;

    public Server() throws IOException {
        serverSocketChannel = AsynchronousServerSocketChannel.open();
    }

    public void start(InetSocketAddress address) throws IOException {
        serverSocketChannel.bind(address);
        serverSocketChannel.accept(null, this);
    }

    @Override
    public void completed(AsynchronousSocketChannel socketChannel, Void struct) {
        System.out.println("有 Client 连接...");
        val client = new Client(socketChannel);
        client.setHandler(frame -> client.write(frame.toByteArray()));
        client.registerRead(1024);
    }

    @Override
    public void failed(Throwable exc, Void struct) {
        exc.printStackTrace();
    }

    @Override
    public void close() throws IOException {
        serverSocketChannel.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        val address = new InetSocketAddress("127.0.0.1", 9999);
        try (val server = new Server()) {
            server.start(address);
            Thread.sleep(500);
            try (val client = new Client()) {
                client.setHandler(frame -> System.out.println(FrameUtil.toString(frame)));
                client.start(address);
                Thread.sleep(500);
                client.write(ByteUtil.fromString("68-04-07-00-00-00 68-04-07-00-00-00"));
                Thread.sleep(500);
            }
        }
    }
}

package demo.IEC104.sc;

import demo.IEC104.FrameUtil;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousServerSocketChannel;
import java.nio.channels.AsynchronousSocketChannel;
import java.nio.channels.CompletionHandler;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@Slf4j
@SuppressWarnings("CallToPrintStackTrace")
public class AsyncSocketServer implements CompletionHandler<AsynchronousSocketChannel, Void>, Closeable {
    private final AsynchronousServerSocketChannel serverSocketChannel;

    public AsyncSocketServer() throws IOException {
        serverSocketChannel = AsynchronousServerSocketChannel.open();
    }

    public void start(InetSocketAddress address) throws IOException {
        serverSocketChannel.bind(address);
        serverSocketChannel.accept(null, this);
    }

    @Override
    public void completed(AsynchronousSocketChannel socketChannel, Void struct) {
        log.debug("有 Client 连接...");
        val asyncSocketClient = new AsyncSocketClient(socketChannel);
        asyncSocketClient.setHandler(AsyncSocketServer::clientHandle);
        asyncSocketClient.registerRead(1024);
    }

    private static boolean clientHandle(AsyncSocketClient client, ByteBuffer buffer) {
        val frame = FrameUtil.parse(buffer);
        if (frame == null) {
            return false;
        }
        client.write(frame.toByteArray());
        return true;
    }

    @Override
    public void failed(Throwable exc, Void struct) {
        exc.printStackTrace();
    }

    @Override
    public void close() {
        try {
            serverSocketChannel.close();
        } catch (IOException e) {
            log.error("AsyncSocketServer 关闭失败", e);
        }
    }
}

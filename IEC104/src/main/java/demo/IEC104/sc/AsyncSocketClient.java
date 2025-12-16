package demo.IEC104.sc;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousSocketChannel;
import java.nio.channels.CompletionHandler;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@Slf4j
@Getter
public class AsyncSocketClient implements Closeable {
    public static final int DEFAULT_BUFFER_SIZE = 1024;
    private final int bufferSize;
    private final AsynchronousSocketChannel socketChannel;
    private volatile boolean reading = false;
    /**
     * 返回true时继续解析，否则停止解析
     */
    @Setter
    private ClientHandler handler;

    public AsyncSocketClient() throws IOException {
        this(DEFAULT_BUFFER_SIZE);
    }

    public AsyncSocketClient(int bufferSize) throws IOException {
        this(bufferSize, AsynchronousSocketChannel.open());
    }

    public AsyncSocketClient(AsynchronousSocketChannel socketChannel) {
        this(DEFAULT_BUFFER_SIZE, socketChannel);
    }

    public AsyncSocketClient(int bufferSize, AsynchronousSocketChannel socketChannel) {
        this.bufferSize = bufferSize;
        this.socketChannel = socketChannel;
    }

    public void start(InetSocketAddress address) {
        socketChannel.connect(address, null, new ConnectedCallback());
    }

    public void registerRead(int size) {
        if (!reading) {
            var buffer = ByteBuffer.allocate(size);
            socketChannel.read(buffer, buffer, new ReadedCallback());
            reading = true;
        }
    }

    public void write(byte[] bf) {
        write(ByteBuffer.wrap(bf));
    }

    public void write(ByteBuffer bf) {
        socketChannel.write(bf, null, new WritedCallback());
    }

    public void parseData(ByteBuffer buffer) {
        while (true) {
            if (handler != null) {
                if (!handler.handle(this, buffer)) {
                    break;
                }
            } else {
                buffer.position(buffer.limit());
                buffer.compact();
                break;
            }
        }
    }

    @Override
    public void close() {
        try {
            if (reading) {
                // 关闭输入，交给read回调内关闭流
                socketChannel.shutdownInput();
            } else {
                socketChannel.close();
            }
        } catch (IOException e) {
            log.error("AsyncSocketClient 关闭失败", e);
        }
    }

    private final class ConnectedCallback implements CompletionHandler<Void, Void> {
        @Override
        public void completed(Void result, Void attachment) {
            log.debug("AsyncSocketClient 已连接...");
            registerRead(bufferSize);
        }

        @Override
        public void failed(Throwable exc, Void attachment) {
            log.error("AsyncSocketClient 连接失败", exc);
        }
    }

    private final class ReadedCallback implements CompletionHandler<Integer, ByteBuffer> {
        @Override
        public void completed(Integer bytesRead, ByteBuffer buffer) {
            if (bytesRead == -1) {
                log.debug("AsyncSocketClient 已断开...");
                reading = false;
                close();
            } else if (bytesRead > 0) {
                // 解析数据
                parseData(buffer);
                // 继续读取
                socketChannel.read(buffer, buffer, this);
            }
        }

        @Override
        public void failed(Throwable exc, ByteBuffer buffer) {
            log.error("AsyncSocketClient 读取数据失败", exc);
        }
    }

    private static class WritedCallback implements CompletionHandler<Integer, Void> {
        @Override
        public void completed(Integer bytesWritten, Void attachment) {
            if (bytesWritten > 0) {
                log.debug("AsyncSocketClient 数据发送成功");
            }
        }

        @Override
        public void failed(Throwable exc, Void attachment) {
            log.error("AsyncSocketClient 发送数据失败", exc);
        }
    }

    public interface ClientHandler {
        boolean handle(AsyncSocketClient client, ByteBuffer frame);
    }
}

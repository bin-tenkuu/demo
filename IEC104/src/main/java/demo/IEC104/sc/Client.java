package demo.IEC104.sc;

import demo.IEC104.ByteUtil;
import demo.IEC104.Frame;
import demo.IEC104.FrameUtil;
import lombok.Setter;
import lombok.val;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.ByteBuffer;
import java.nio.channels.AsynchronousSocketChannel;
import java.nio.channels.CompletionHandler;
import java.util.function.Consumer;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@SuppressWarnings("CallToPrintStackTrace")
public class Client implements CompletionHandler<Integer, ByteBuffer>, Closeable {
    private final AsynchronousSocketChannel socketChannel;
    private volatile boolean reading = false;
    @Setter
    private Consumer<Frame> handler;

    public Client() throws IOException {
        this(AsynchronousSocketChannel.open());
    }

    public Client(AsynchronousSocketChannel socketChannel) {
        this.socketChannel = socketChannel;
    }

    public void start(InetSocketAddress address) throws IOException {
        socketChannel.connect(address, null, new CompletionHandler<Void, Void>() {
            @Override
            public void completed(Void result, Void attachment) {
                System.out.println("Client 已连接...");
                registerRead(1024);
            }

            @Override
            public void failed(Throwable exc, Void attachment) {
                exc.printStackTrace();
            }
        });
    }

    public void registerRead(int size) {
        if (!reading) {
            val buffer = ByteBuffer.allocate(size);
            socketChannel.read(buffer, buffer, Client.this);
            reading = true;
        }
    }

    @Override
    public void completed(Integer bytesRead, ByteBuffer buffer) {
        if (bytesRead == -1) {
            System.out.println("Client 已断开...");
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
        exc.printStackTrace();
    }

    public void write(byte[] bf) {
        write(ByteBuffer.wrap(bf));
    }

    public void write(ByteBuffer bf) {
        socketChannel.write(bf, null, new CompletionHandler<>() {
            @Override
            public void completed(Integer bytesWritten, Object attachment) {
                if (bytesWritten > 0) {
                    System.out.println("Client 数据发送成功");
                }
            }

            @Override
            public void failed(Throwable exc, Object attachment) {
                exc.printStackTrace();
            }
        });
    }

    private void parseData(ByteBuffer buffer) {
        while (true) {
            buffer.flip();
            var remaining = buffer.remaining();
            if (remaining > 2) {
                var length = buffer.get(1) + 2;
                if (remaining >= length) {
                    var bs = new byte[length];
                    val pos = buffer.position();
                    buffer.get(pos, bs, 0, length);
                    buffer.position(pos + length);
                    buffer.compact();
                    if (handler != null) {
                        handler.accept(FrameUtil.parse(bs));
                    }
                    continue;
                }
            }
            buffer.compact();
            break;
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
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        Thread.ofVirtual().start(() -> {
            try {
                try (val serverSocket = new ServerSocket(9999)) {
                    try (val socket = serverSocket.accept()) {
                        // redirect
                        try (val in = socket.getInputStream();
                             val out = socket.getOutputStream()) {
                            int read;
                            while (!socket.isClosed() && (read = in.read()) >= 0) {
                                out.write(read);
                            }
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        Thread.sleep(500);
        try (val client = new Client()) {
            client.setHandler(frame -> System.out.println(FrameUtil.toString(frame)));
            client.start(new InetSocketAddress("127.0.0.1", 9999));
            Thread.sleep(500);
            client.write(ByteUtil.fromString("68-04-07-00-00-00 68-04-07-00-00-00"));
            Thread.sleep(500);
        }
    }
}

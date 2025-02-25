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
            socketChannel.read(buffer, buffer, this);
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

    public void write(Frame frame) {
        write(ByteBuffer.wrap(frame.toByteArray()));
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
            val frame = FrameUtil.parse(buffer);
            if (frame == null) {
                break;
            }
            if (handler != null) {
                handler.accept(frame);
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
            val bs = ByteUtil.fromString("""
                    68 d5 cc 07 fc 20 0f a8 25 00 01 00 01 64 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
                    a1 0d 00 00 00 be 01 00 00 00 5f 0f 00 00 00 8e 05 00 00 00 c0 03 00 00 00 05 00 00 00 00 c5 03 00 00 00
                    15 03 00 00 00 0b 00 00 00 00 94 01 00 00 00 9f 01 00 00 00 88 01 00 00 00 05 00 00 00 00 67 00 00 00 00
                    6d 00 00 00 00 4b 00 00 00 00 d9 02 00 00 00 1b 00 00 00 00 f4 02 00 00 00 dd 01 00 00 00 00 00 00 00 00
                    00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 29 0a 00 00 00 85 01 00 00 00 ae 0b 00 00 00 79 09 00 00 00
                    3b 06 00 00 00 11 00 00 00 00 4d 06 00 00 00 cf 03 00 00 00 d7 10 00 00 00 8e 01 00 00 00 66 12 00 00 00
                    c0 08 00 00 00""");
            client.write(bs);
            Thread.sleep(500);
        }
    }
}

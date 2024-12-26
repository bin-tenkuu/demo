package demo.IEC104.sc;

import lombok.val;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.SocketChannel;
import java.util.Iterator;
import java.util.Set;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@SuppressWarnings("CallToPrintStackTrace")
public class Client implements Runnable, Closeable {
    private final SocketChannel socket;
    private final Selector selector;
    private final Thread thread;

    public Client(InetSocketAddress address) throws IOException {
        // 初始化客户端
        socket = SocketChannel.open();
        socket.configureBlocking(false);
        selector = Selector.open();
        // 注册连接事件
        socket.register(selector, SelectionKey.OP_CONNECT);
        // 发起连接
        socket.connect(address);
        System.out.println("客户端开启了...");
        thread = new Thread(this);
        thread.start();
    }

    @Override
    public void run() {
        try {
            while (socket.isOpen()) {
                selector.select();
                Set<SelectionKey> keys = selector.selectedKeys();
                Iterator<SelectionKey> iterator = keys.iterator();
                // 处理准备就绪的事件
                while (iterator.hasNext()) {
                    SelectionKey key = iterator.next();
                    // 删除当前键，避免重复消费
                    iterator.remove();

                    handleKey(key);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("客户端异常，请重启！");
        }
    }

    public void write(ByteBuffer bf) throws IOException {
        socket.write(bf);
        // // 用户已输入，注册写事件，将输入的消息发送给客户端
        // socket.register(selector, SelectionKey.OP_WRITE, bf);
        // // 唤醒之前因为监听OP_READ而阻塞的select()
        // selector.wakeup();
    }

    private void handleKey(SelectionKey key) throws IOException {
        if (key.isConnectable()) {
            // 在非阻塞模式下connect也是非阻塞的，所以要确保连接已经建立完成
            while (!socket.finishConnect()) {
                System.out.println("连接中");
            }
            System.out.println("服务端已连接");
            socket.register(selector, SelectionKey.OP_READ, ByteBuffer.allocate(1024));
        }
        // 控制台监听到有输入，注册OP_WRITE,然后将消息附在attachment中
        if (key.isWritable()) {
            // 发送消息给服务端
            // socket.write((ByteBuffer) key.attachment());
        }
        // 处理输入事件
        if (key.isReadable()) {
            val bf = (ByteBuffer) key.attachment();
            val bs = new byte[4];
            int len;
            while ((len = socket.read(bf)) > 0) {
                if (bf.limit() < 4) {
                    System.out.println("bf.limit() < 4");
                    break;
                }
                bf.get(bs);
                for (byte b : bs) {
                    System.out.print(b);
                    System.out.print(' ');
                }
                System.out.println();
            }
            if (len == -1) {
                key.cancel();
                socket.close();
                System.out.println("客戶端已关闭...");
            }
        }
    }

    @Override
    public void close() throws IOException {
        thread.interrupt();
        socket.close();
        selector.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        try (val client = new Client(new InetSocketAddress(9999))) {
            byte[] bs = new byte[]{
                    1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0
            };
            Thread.sleep(500);
            val bf = ByteBuffer.allocate(1);
            for (int i = 0; i < 4; i++) {
                for (byte b : bs) {
                    bf.put(0, b);
                    client.write(bf);
                    Thread.sleep(500);
                }
            }
        }
    }
}

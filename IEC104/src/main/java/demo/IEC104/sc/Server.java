package demo.IEC104.sc;

import lombok.AllArgsConstructor;
import lombok.val;

import java.io.Closeable;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.time.LocalDateTime;
import java.util.Iterator;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@SuppressWarnings("CallToPrintStackTrace")
@AllArgsConstructor
public class Server implements Runnable, Closeable {
    private final ServerSocketChannel serverSocket;
    private final Selector selector;
    private final Thread thread;

    public Server(InetSocketAddress address) throws IOException {
        // 服务初始化
        serverSocket = ServerSocketChannel.open();
        // 设置为非阻塞
        serverSocket.configureBlocking(false);
        // 绑定端口
        serverSocket.bind(address);
        // 注册OP_ACCEPT事件（即监听该事件，如果有客户端发来连接请求，则该键在select()后被选中）
        selector = Selector.open();
        serverSocket.register(selector, SelectionKey.OP_ACCEPT);
        System.out.println("服务端开启了...");
        thread = new Thread(this);
        thread.start();
    }

    @SuppressWarnings("InfiniteLoopStatement")
    @Override
    public void run() {
        try {
            // 轮询服务
            while (true) {
                selector.select();
                Iterator<SelectionKey> it = selector.selectedKeys().iterator();
                while (it.hasNext()) {
                    SelectionKey key = it.next();
                    // 处理掉后将键移除，避免重复消费(因为下次选择后，还在已选择键集中)
                    it.remove();
                    handleKey(key);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("服务器异常，关闭...");
        }

    }

    private void handleKey(SelectionKey key) throws IOException {
        if (key.isAcceptable()) {
            // 接收客户端连接
            SocketChannel socket = serverSocket.accept();
            socket.configureBlocking(false);
            socket.register(selector, SelectionKey.OP_READ, ByteBuffer.allocate(1024));
            // 输出客户端地址
            System.out.println(LocalDateTime.now() + "\t" + socket.getRemoteAddress().toString());
            System.out.println("客戶端已连接...");
        }
        if (key.isReadable()) {
            // 读取数据
            SocketChannel socket = (SocketChannel) key.channel();
            val bf = (ByteBuffer) key.attachment();
            int len;
            while ((len = socket.read(bf)) > 0) {
                System.out.printf("转发 %s 字节\n" , bf.limit());
                socket.write(bf);
            }
            if (len == -1) {
                // 客户端关闭了
                key.cancel();
                socket.close();
                System.out.println("客戶端已断开...");
            }
        }
    }

    @Override
    public void close() throws IOException {
        thread.interrupt();
        serverSocket.close();
        selector.close();
    }

    public static void main(String[] args) throws IOException {
        new Server(new InetSocketAddress(9999));
    }
}

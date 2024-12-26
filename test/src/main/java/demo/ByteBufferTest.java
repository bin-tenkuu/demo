package demo;

import lombok.AllArgsConstructor;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.Calendar;
import java.util.Iterator;
import java.util.Scanner;
import java.util.Set;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/11/12
 */
@SuppressWarnings("CallToPrintStackTrace")
public class ByteBufferTest {
    public static void main(String[] args) {
        try {
            // 初始化客户端
            SocketChannel socket = SocketChannel.open();
            socket.configureBlocking(false);
            Selector selector = Selector.open();
            // 注册连接事件
            socket.register(selector, SelectionKey.OP_CONNECT);
            // 发起连接
            socket.connect(new InetSocketAddress("127.0.0.1", 9999));
            // 开启控制台输入监听
            new ChatThread(selector, socket).start();
            Calendar ca = Calendar.getInstance();
            // 轮询处理
            while (socket.isOpen()) {
                // 在注册的键中选择已准备就绪的事件
                selector.select();
                // 已选择键集
                Set<SelectionKey> keys = selector.selectedKeys();
                Iterator<SelectionKey> iterator = keys.iterator();
                // 处理准备就绪的事件
                while (iterator.hasNext()) {
                    SelectionKey key = iterator.next();
                    // 删除当前键，避免重复消费
                    iterator.remove();
                    // 连接
                    if (key.isConnectable()) {
                        // 在非阻塞模式下connect也是非阻塞的，所以要确保连接已经建立完成
                        while (!socket.finishConnect()) {
                            System.out.println("连接中");
                        }
                        socket.register(selector, SelectionKey.OP_READ);
                    }
                    // 控制台监听到有输入，注册OP_WRITE,然后将消息附在attachment中
                    if (key.isWritable()) {
                        // 发送消息给服务端
                        socket.write((ByteBuffer) key.attachment());
                        /*
                            已处理完此次输入，但OP_WRITE只要当前通道输出方向没有被占用
                            就会准备就绪，select()不会阻塞（但我们需要控制台触发,在没有输入时
                            select()需要阻塞），因此改为监听OP_READ事件，该事件只有在socket
                            有输入时select()才会返回。
                        */
                        socket.register(selector, SelectionKey.OP_READ);
                        System.out.println("==============" + ca.getTime() + " ==============");
                    }
                    // 处理输入事件
                    if (key.isReadable()) {

                        ByteBuffer byteBuffer = ByteBuffer.allocate(1024 * 4);
                        int len;
                        // 捕获异常，因为在服务端关闭后会发送FIN报文，会触发read事件，但连接已关闭,此时read()会产生异常
                        try {
                            if ((len = socket.read(byteBuffer)) > 0) {
                                System.out.println("接收到來自服务器的消息\t");
                                System.out.println(new String(byteBuffer.array(), 0, len));
                            }
                        } catch (IOException e) {
                            System.out.println("服务器异常，请联系客服人员!正在关闭客户端.........");
                            key.cancel();
                            socket.close();
                        }
                        System.out.println("=========================================================");
                    }
                }
            }

        } catch (IOException e) {
            System.out.println("客户端异常，请重启！");
        }
    }

    public static class ServerSocket {
        public static void main(String[] args) {
            try {
                // 服务初始化
                ServerSocketChannel serverSocket = ServerSocketChannel.open();
                // 设置为非阻塞
                serverSocket.configureBlocking(false);
                // 绑定端口
                serverSocket.bind(new InetSocketAddress("127.0.0.1", 9999));
                // 注册OP_ACCEPT事件（即监听该事件，如果有客户端发来连接请求，则该键在select()后被选中）
                Selector selector = Selector.open();
                serverSocket.register(selector, SelectionKey.OP_ACCEPT);
                Calendar ca = Calendar.getInstance();
                System.out.println("服务端开启了...");
                // 轮询服务
                while (true) {
                    // 选择准备好的事件
                    selector.select();
                    // 已选择的键集
                    Iterator<SelectionKey> it = selector.selectedKeys().iterator();
                    // 处理已选择键集事件
                    while (it.hasNext()) {
                        SelectionKey key = it.next();
                        // 处理掉后将键移除，避免重复消费(因为下次选择后，还在已选择键集中)
                        it.remove();
                        // 处理连接请求
                        if (key.isAcceptable()) {
                            // 处理请求
                            SocketChannel socket = serverSocket.accept();
                            socket.configureBlocking(false);
                            // 注册read，监听客户端发送的消息
                            socket.register(selector, SelectionKey.OP_READ);
                            // keys为所有键，除掉serverSocket注册的键就是已连接socketChannel的数量
                            String message = "连接成功 你是第" + (selector.keys().size() - 1) + "个用户";
                            // 向客户端发送消息
                            socket.write(ByteBuffer.wrap(message.getBytes()));
                            InetSocketAddress address = (InetSocketAddress) socket.getRemoteAddress();
                            // 输出客户端地址
                            System.out.println(ca.getTime() + "\t" + address.getHostString() +
                                               ":" + address.getPort() + "\t");
                            System.out.println("客戶端已连接...");
                        }

                        if (key.isReadable()) {
                            SocketChannel socket = (SocketChannel) key.channel();
                            InetSocketAddress address = (InetSocketAddress) socket.getRemoteAddress();
                            System.out.println(ca.getTime() + "\t" + address.getHostString() +
                                               ":" + address.getPort() + "\t");
                            ByteBuffer bf = ByteBuffer.allocate(1024 * 4);
                            int len;
                            byte[] res = new byte[1024 * 4];
                            // 捕获异常，因为在客户端关闭后会发送FIN报文，会触发read事件，但连接已关闭,此时read()会产生异常
                            try {
                                while ((len = socket.read(bf)) != 0) {
                                    bf.flip();
                                    bf.get(res, 0, len);
                                    System.out.println(new String(res, 0, len));
                                    bf.clear();
                                }
                            } catch (IOException e) {
                                // 客户端关闭了
                                key.cancel();
                                socket.close();
                                System.out.println("客戶端已断开");
                            }
                        }
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("服务器异常，即将关闭..........");
            }
        }
    }

    @AllArgsConstructor
    public static class ChatThread extends Thread {

        private final Selector selector;
        private final SocketChannel socket;

        @Override
        public void run() {
            try {
                // 等待连接建立
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            Scanner scanner = new Scanner(System.in);
            System.out.println("请输入您要发送给服务端的消息");
            System.out.println("=========================================================");
            while (scanner.hasNextLine()) {
                String s = scanner.nextLine();
                try {
                    // 用户已输入，注册写事件，将输入的消息发送给客户端
                    socket.register(selector, SelectionKey.OP_WRITE, ByteBuffer.wrap(s.getBytes()));
                    // 唤醒之前因为监听OP_READ而阻塞的select()
                    selector.wakeup();
                } catch (ClosedChannelException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}

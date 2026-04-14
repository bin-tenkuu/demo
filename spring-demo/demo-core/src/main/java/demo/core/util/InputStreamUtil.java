package demo.core.util;

import java.io.*;
import java.nio.charset.Charset;

/**
 * @author bin
 * @since 2026/03/19
 */
public class InputStreamUtil implements Closeable {
    private final InputStream in;
    private final byte[] bytes;
    private int index = 0;
    private int limit;

    public InputStreamUtil(InputStream in) {
        this(in, 1024);
    }

    public InputStreamUtil(InputStream in, int bufferSize) {
        this.in = new BufferedInputStream(in);
        this.bytes = new byte[bufferSize];
    }

    @Override
    public void close() throws IOException {
        in.close();
    }

    public void read(int size) throws IOException {
        if (limit - index >= size) {
            // 如果缓存大于需要的字节，直接返回
            return;
        }
        System.arraycopy(bytes, index, bytes, 0, limit - index);
        limit = limit - index;
        index = 0;
        int read;
        do {
            read = in.read(bytes, limit, bytes.length - limit);
            if (read == -1) {
                throw new EOFException("读取到文件末尾");
            }
            limit += read;
        } while (limit < size);
    }

    public String getString(int size, Charset charset) {
        var s = ByteUtil.getString(bytes, index, size, charset);
        index += size;
        return s;
    }

    public int getInt() {
        var i = ByteUtil.getInt(bytes, index);
        index += 4;
        return i;
    }

    public float getFloat() {
        var f = ByteUtil.getFloat(bytes, index);
        index += 4;
        return f;
    }

}

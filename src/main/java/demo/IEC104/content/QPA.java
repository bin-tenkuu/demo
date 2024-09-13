package demo.IEC104.content;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/13
 */
public class QPA implements BaseContent {
    private byte value;

    public QPA(byte b) {
        value = b;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = value;
    }

    @Override
    public String toString() {
        return "激活/停止(QPA)=" + value;
    }
}

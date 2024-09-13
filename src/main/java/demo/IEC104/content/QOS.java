package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/13
 */
@Getter
@Setter
public class QOS implements BaseContent {
    private boolean se;
    private int value;

    public QOS(byte b) {
        se = ByteUtil.getBit(b, 7);
        value = b & 0xFF << 1 >>> 1;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = 0;
        b = ByteUtil.setBit(b, 7, se);
        data[offset] = (byte) (b | (value & 0x7f));
    }

    @Override
    public String toString() {
        return "选择/执行(S/E)=" + se +
                ", 命令(QOS)=" + value;
    }
}

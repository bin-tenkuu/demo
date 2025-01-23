package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

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
        value = (b & 0xFF) << 1 >>> 1;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = (byte) (value & 0b0111_1111);
        b = ByteUtil.setBit(b, 7, se);
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("选择/执行(S/E)=").append(se ? "选择" : "执行")
                .append("，命令=").append(value);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

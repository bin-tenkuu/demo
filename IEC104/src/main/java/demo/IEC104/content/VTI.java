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
public class VTI implements BaseContent {
    private boolean t;
    private int value;

    public VTI(byte b) {
        t = ByteUtil.getBit(b, 7);
        value = (b & 0xfe) >> 1;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = ByteUtil.setBit((byte) (value & 0x7f), 7, t);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("是否瞬变=").append(t ? "是" : "否")
                .append("，数据=").append(value);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

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
public class QPM implements BaseContent {
    private boolean isRunning;
    private boolean isChange;
    private int type;

    public QPM(byte b) {
        isRunning = ByteUtil.getBit(b, 7);
        isChange = ByteUtil.getBit(b, 6);
        type = b & 0x3f;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = 0;
        b = ByteUtil.setBit(b, 7, isRunning);
        b = ByteUtil.setBit(b, 6, isChange);
        data[offset] = (byte) (b | type & 0x3f);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("是否运行=").append(isRunning ? "是" : "否")
                .append("，是否改变=").append(isChange ? "是" : "否")
                .append("，类型=").append(type);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

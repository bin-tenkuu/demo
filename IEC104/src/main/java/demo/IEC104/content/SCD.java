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
public class SCD implements BaseContent {
    private int value;

    public SCD(byte[] content, int offset) {
        value = ByteUtil.getInt(content, offset);
    }

    @Override
    public int size() {
        return 4;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setInt(data, offset, value);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("数据=").append(value);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/12
 */
@Getter
@Setter
public class IOA implements BaseContent {
    private short addr;

    public IOA(byte[] content, int offset) {
        addr = ByteUtil.getShort(content, offset);
    }

    @Override
    public int size() {
        return 3;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setShort(data, offset, addr);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("地址=").append(addr);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

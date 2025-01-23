package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/23
 */
@Getter
@Setter
public class TSC implements BaseContent {
    private short tsc;

    public TSC(byte[] data, int offset) {
        tsc = ByteUtil.getShort(data, offset);
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setShort(data, offset, tsc);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("TSC=").append(tsc);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

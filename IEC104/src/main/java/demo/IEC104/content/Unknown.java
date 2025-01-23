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
public class Unknown implements BaseContent {
    private byte b;

    public Unknown(byte b) {
        this.b = b;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append(ByteUtil.toString(b));
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

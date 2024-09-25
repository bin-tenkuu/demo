package demo.IEC104.content;

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
    public void toString(StringBuilder builder) {
        builder.append("参数=").append(value);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

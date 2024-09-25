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
public class QOI implements BaseContent {
    private byte value;

    public QOI(byte value) {
        this.value = value;
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
        builder.append("QOI=").append(value);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

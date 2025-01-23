package demo.IEC104.content;

import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/23
 */
@Getter
@Setter
public class CHS implements BaseContent {
    private int chs;

    public CHS(byte b) {
        chs = b & 0xFF;
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) (chs);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("CHS=").append(chs);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

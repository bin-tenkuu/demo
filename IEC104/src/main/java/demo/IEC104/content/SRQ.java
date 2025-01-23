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
public class SRQ implements BaseContent {
    private boolean bsi;

    public SRQ(byte b) {
        bsi = ByteUtil.getBit(b, 7);
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) (bsi ? 0x80 : 0);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("节准备就绪去装载(BSI)=").append(bsi ? "否" : "是");
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

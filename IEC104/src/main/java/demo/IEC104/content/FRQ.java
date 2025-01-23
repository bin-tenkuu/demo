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
public class FRQ implements BaseContent {
    private boolean bsi;

    public FRQ(byte b) {
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
        builder.append("选择、请求、停止激活或删除的确认(BSI)=").append(bsi ? "否定" : "肯定");
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

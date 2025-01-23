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
public class BCR implements BaseContent {
    private boolean iv;
    private boolean ca;
    private boolean cy;
    private int sq;
    private int bcr;

    public BCR(byte[] data, int offset) {
        byte b = data[offset];
        iv = ByteUtil.getBit(b, 7);
        ca = ByteUtil.getBit(b, 6);
        cy = ByteUtil.getBit(b, 5);
        sq = b & 0x1f;
        bcr = ByteUtil.getInt(data, offset + 1);
    }

    @Override
    public int size() {
        return 5;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = 0;
        b = ByteUtil.setBit(b, 7, iv);
        b = ByteUtil.setBit(b, 6, ca);
        b = ByteUtil.setBit(b, 5, cy);
        b = (byte) (b | sq & 0x1f);
        data[offset] = b;
        ByteUtil.setInt(data, offset + 1, bcr);
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("IV=").append(iv ? "无效" : "有效")
                .append("，计数量被调整(CA)=").append(ca ? "已调整" : "未调整")
                .append("，进位(CY)=").append(cy ? "已溢出" : "未溢出")
                .append("，顺序号(SQ)=").append(sq)
                .append("，计数量读数)=").append(bcr);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

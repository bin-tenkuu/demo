package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;

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
    public String toString() {
        return "是否无效(IV)=" + iv +
                ", 是否确认(CA)=" + ca +
                ", 是否同步(CY)=" + cy +
                ", 传输序号(SQ)=" + sq +
                ", 传输原因(BCR)=" + bcr;
    }
}

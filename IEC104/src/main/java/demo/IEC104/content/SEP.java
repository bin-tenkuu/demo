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
public class SEP implements BaseContent {
    private boolean srd;
    private boolean sie;
    private boolean sl3;
    private boolean sl2;
    private boolean sl1;
    private boolean gs;

    public SEP(byte b) {
        srd = ByteUtil.getBit(b, 5);
        sie = ByteUtil.getBit(b, 4);
        sl3 = ByteUtil.getBit(b, 3);
        sl2 = ByteUtil.getBit(b, 2);
        sl1 = ByteUtil.getBit(b, 1);
        gs = ByteUtil.getBit(b, 0);
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = 0;
        b = ByteUtil.setBit(b, 5, srd);
        b = ByteUtil.setBit(b, 4, sie);
        b = ByteUtil.setBit(b, 3, sl3);
        b = ByteUtil.setBit(b, 2, sl2);
        b = ByteUtil.setBit(b, 1, sl1);
        b = ByteUtil.setBit(b, 0, gs);
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("反向保护启动(SRD)=").append(srd ? "是" : "否")
                .append("，接地电流保护启动(SIE)=").append(sie ? "是" : "否")
                .append("，C相保护启动(SL3)=").append(sl3 ? "是" : "否")
                .append("，B相保护启动(SL2)=").append(sl2 ? "是" : "否")
                .append("，A相保护启动(SL1)=").append(sl1 ? "是" : "否")
                .append("，总启动(GS)=").append(gs ? "是" : "否");
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

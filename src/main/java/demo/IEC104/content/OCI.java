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
public class OCI implements BaseContent {
    private boolean sl3;
    private boolean sl2;
    private boolean sl1;
    private boolean gs;

    public OCI(byte b) {
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
        b = ByteUtil.setBit(b, 3, sl3);
        b = ByteUtil.setBit(b, 2, sl2);
        b = ByteUtil.setBit(b, 1, sl1);
        b = ByteUtil.setBit(b, 0, gs);
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("C相保护启动(SL3)=").append(sl3)
                .append("，B相保护启动(SL2)=").append(sl2)
                .append("，A相保护启动(SL1)=").append(sl1)
                .append("，总启动(GS)=").append(gs);
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

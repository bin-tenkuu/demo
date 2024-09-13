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
public class Command implements BaseContent {
    private boolean se;
    private int co;
    private int cs;

    public Command(byte b) {
        se = ByteUtil.getBit(b, 7);
        co = b & 0xFF << 1 >>> 3;
        cs = b & 3;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = ByteUtil.setBit((byte) ((co & 0x1f) << 2 | (cs & 3)), 7, se);
    }

    @Override
    public String toString() {
        return "选择/执行(S/E)=" + se +
                ", 命令序号(CO)=" + co +
                ", 命令状态(CS)=" + cs;
    }
}

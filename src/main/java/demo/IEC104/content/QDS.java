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
public class QDS extends BaseQds implements BaseContent {
    protected boolean ov;

    public QDS(byte b) {
        super(b);
        ov = ByteUtil.getBit(b, 0);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] = ByteUtil.setBit(data[offset], 0, ov);
    }

    @Override
    public String toString() {
        return super.toString() +
                ", 是否溢出(OV)=" + ov;
    }
}

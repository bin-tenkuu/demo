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
public class NVA implements BaseContent {
    private float value;

    public NVA(byte[] data, int offset) {
        value = ByteUtil.getShort(data, offset) / 32768F;
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setShort(data, offset, (short) (value * 32768));
    }

    @Override
    public String toString() {
        return super.toString() +
                ", 归一值(NVA)=" + value;
    }
}

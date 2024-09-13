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
public class SVA implements BaseContent {
    protected short value;

    public SVA(byte[] data, int offset) {
        value = ByteUtil.getShort(data, offset);
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setShort(data, offset, value);
    }

    @Override
    public String toString() {
        return "标量值(SVA)=" + value;
    }
}

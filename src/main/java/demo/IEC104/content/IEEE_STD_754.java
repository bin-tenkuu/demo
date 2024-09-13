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
public class IEEE_STD_754 implements BaseContent {
    private float value;

    public IEEE_STD_754(byte[] content, int offset) {
        value = ByteUtil.getFloat(content, offset);
    }

    @Override
    public int size() {
        return 4;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setFloat(data, offset, value);
    }

    @Override
    public String toString() {
        return "数据(IEEE_STD_754)=" + value;
    }
}

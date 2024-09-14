package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/12
 */
@Getter
@Setter
public class Unknown implements BaseContent {

    private byte[] content;
    private String name;

    public Unknown(byte[] content, int offset, int length, String name) {
        val bs = new byte[length];
        System.arraycopy(content, offset, bs, 0, length);
        this.content = bs;
        this.name = name;
    }

    @Override
    public int size() {
        return content.length;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        System.arraycopy(content, 0, data, offset, content.length);
    }

    @Override
    public String toString() {
        return name + "=" + ByteUtil.toString(content);
    }
}

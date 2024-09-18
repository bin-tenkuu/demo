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
public class SIQ extends BaseQds implements BaseContent {
    protected boolean value;

    public SIQ(byte b) {
        super(b);
        value = ByteUtil.getBit(b, 0);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] = ByteUtil.setBit(data[offset], 0, value);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append("，单点遥信=").append(value ? "合" : "分");
    }
}

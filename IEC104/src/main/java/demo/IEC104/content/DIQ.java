package demo.IEC104.content;

import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/13
 */
@Getter
@Setter
public class DIQ extends BaseQds implements BaseContent {
    protected int value;

    public DIQ(byte b) {
        super(b);
        value = b & 3;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] |= (byte) (value & 3);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        String type = switch (value) {
            case 1 -> "分";
            case 2 -> "合";
            default -> "不确定";
        };
        builder.append("，双点遥信=").append(type).append("(").append(value).append(")");
    }
}

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
    private int dpi;

    public DIQ(byte b) {
        super(b);
        dpi = b & 3;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] |= (byte) (dpi & 3);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        String type = switch (dpi) {
            case 1 -> "开";
            case 2 -> "合";
            default -> "不确定或中间状态";
        };
        builder.append("，双点信息(dpi)=").append(type).append("(").append(dpi).append(")");
    }
}

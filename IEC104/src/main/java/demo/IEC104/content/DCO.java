package demo.IEC104.content;

import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/23
 */
@Getter
@Setter
public class DCO extends QOC implements BaseContent {
    private int dcs;

    public DCO(byte b) {
        super(b);
        dcs = (b & 3);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) (data[offset] | dcs & 3);
    }

    public static String byDCS(int dcs) {
        return switch (dcs) {
            case 1 -> "开";
            case 2 -> "合";
            default -> "不允许";
        };
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append(", 双命令状态=").append(byDCS(dcs));
    }

}

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
public class RCO extends QOS implements BaseContent {
    private int rcs;

    public RCO(byte b) {
        super(b);
        rcs = (b & 3);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) (data[offset] | rcs & 3);
    }

    public static String byRCS(int dcs) {
        return switch (dcs) {
            case 1 -> "下一步降";
            case 2 -> "下一步升";
            default -> "不允许";
        };
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append(", 双命令状态=").append(byRCS(rcs));
    }

}

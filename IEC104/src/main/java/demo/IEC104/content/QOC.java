package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/23
 */
@Getter
@Setter
public class QOC implements BaseContent {
    private boolean se;
    private int qu;

    public QOC(byte b) {
        se = ByteUtil.getBit(b, 7);
        qu = (b & 0b0111_1100) >> 2;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = (byte) (qu & 0b0111_1100);
        b = ByteUtil.setBit(b, 7, se);
        data[offset] = b;
    }

    public static String byQU(int qu) {
        if (qu <= 3) {
            return switch (qu) {
                case 0 -> "无另外的定义";
                case 1 -> "短脉冲持续时间(断路器)";
                case 2 -> "长脉冲持续时间";
                case 3 -> "持续输出";
                default -> throw new IllegalStateException("Unexpected value: " + qu);
            };
        } else if (qu <= 8) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else if (qu <= 15) {
            return "为其他预先定义的功能选集保留";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("S/E=").append(se ? "选择" : "执行")
                .append("，QU=").append(byQU(qu))
        ;
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

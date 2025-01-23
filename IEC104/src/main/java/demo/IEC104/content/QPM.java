package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/13
 */
@Getter
@Setter
public class QPM implements BaseContent {
    private boolean pop;
    private boolean lpc;
    private int kpa;

    public QPM(byte b) {
        pop = ByteUtil.getBit(b, 7);
        lpc = ByteUtil.getBit(b, 6);
        kpa = b & 0x3f;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = (byte) (kpa & 0b0011_1111);
        b = ByteUtil.setBit(b, 7, pop);
        b = ByteUtil.setBit(b, 6, lpc);
        data[offset] = b;
    }

    public static String byKPA(int kpa) {
        if (kpa <= 4) {
            return switch (kpa) {
                case 0 -> "未用";
                case 1 -> "门限值";
                case 2 -> "平滑系数(滤波时间常数)";
                case 3 -> "测量值传送的上限";
                case 4 -> "测量值传送的下限";
                default -> throw new IllegalStateException("Unexpected value: " + kpa);
            };
        } else if (kpa <= 31) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("参数在运行(POP)=").append(pop ? "是" : "否")
                .append("，当地参数改变(LPC)=").append(lpc ? "是" : "否")
                .append("，参数类别(KPA)=").append(byKPA(kpa));
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

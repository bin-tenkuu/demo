package demo.IEC104.content;

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
public class QCC implements BaseContent {
    private int frz;
    private int rqt;

    public QCC(byte b) {
        frz = (b & 0b1100_0000) >> 6;
        rqt = b & 0b0011_1111;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = (byte) ((frz & 3) << 6 | (rqt & 0b0011_1111));
        data[offset] = b;
    }

    public static String byFRZ(int frz) {
        return switch (frz) {
            case 0 -> "读（无冻结和复位）";
            case 1 -> "计数量冻结不带复位（被冻结的值代表累计）";
            case 2 -> "计数量冻结带复位（被冻结的值代表增量信息）";
            case 3 -> "计数量复位";
            default -> throw new IllegalStateException("Unexpected value: " + frz);
        };
    }

    public static String byRQT(int rqt) {
        if (rqt <= 5) {
            return switch (rqt) {
                case 0 -> "无请求计数量（未采用）";
                case 1 -> "请求计数量第 1 组";
                case 2 -> "请求计数量第 2 组";
                case 3 -> "请求计数量第 3 组";
                case 4 -> "请求计数量第 4 组";
                case 5 -> "请求计数量第 5 组";
                default -> throw new IllegalStateException("Unexpected value: " + rqt);
            };
        } else if (rqt <= 31) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("FRZ=").append(byFRZ(frz))
                .append("，RQT=").append(byRQT(rqt))
        ;
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

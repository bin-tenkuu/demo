package demo.IEC104.content;

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
public class QOI implements BaseContent {
    private byte qoi;

    public QOI(byte b) {
        this.qoi = b;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = qoi;
    }

    public static String byQOI(int qoi) {
        if (qoi == 0) {
            return "未用";
        } else if (qoi <= 19) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else if (qoi <= 36) {
            return switch (qoi) {
                case 20 -> "响应站召唤";
                case 21 -> "响应第 1 组召唤";
                case 22 -> "响应第 2 组召唤";
                case 23 -> "响应第 3 组召唤";
                case 24 -> "响应第 4 组召唤";
                case 25 -> "响应第 5 组召唤";
                case 26 -> "响应第 6 组召唤";
                case 27 -> "响应第 7 组召唤";
                case 28 -> "响应第 8 组召唤";
                case 29 -> "响应第 9 组召唤";
                case 30 -> "响应第 10 组召唤";
                case 31 -> "响应第 11 组召唤";
                case 32 -> "响应第 12 组召唤";
                case 33 -> "响应第 13 组召唤";
                case 34 -> "响应第 14 组召唤";
                case 35 -> "响应第 15 组召唤";
                case 36 -> "响应第 16 组召唤";
                default -> throw new IllegalStateException("Unexpected value: " + qoi);
            };
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("QOI=").append(byQOI(qoi));
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

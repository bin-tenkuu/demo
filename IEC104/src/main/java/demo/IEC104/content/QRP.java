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
public class QRP implements BaseContent {
    private int qrp;

    public QRP(byte b) {
        qrp = b & 0xFF;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) qrp;
    }

    public static String byQRP(int qrp) {
        if (qrp <= 2) {
            return switch (qrp) {
                case 0 -> "未采用";
                case 1 -> "进程的总复位";
                case 2 -> "复位事件缓冲区等待处理的带时标的信息";
                default -> throw new IllegalStateException("Unexpected value: " + qrp);
            };
        } else if (qrp <= 31) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("QRP=").append(byQRP(qrp));
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

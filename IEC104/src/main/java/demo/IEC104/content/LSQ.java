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
public class LSQ implements BaseContent {
    private int lsq;

    public LSQ(byte b) {
        lsq = b & 0xFF;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) (lsq);
    }

    public static String byLSQ(int lsq) {
        if (lsq <= 4) {
            return switch (lsq) {
                case 0 -> "未用";
                case 1 -> "不带停止停止激活的文件传输";
                case 2 -> "带停止激活的文件传输";
                case 3 -> "不带停止激活的节传输";
                case 4 -> "带停止激活的节传输";
                default -> throw new IllegalStateException("Unexpected value: " + lsq);
            };
        } else if (lsq <= 10) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("LSQ=").append(byLSQ(lsq))
        ;
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

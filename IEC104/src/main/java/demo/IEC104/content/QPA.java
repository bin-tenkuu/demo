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
public class QPA implements BaseContent {
    private byte qpa;

    public QPA(byte b) {
        qpa = b;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = qpa;
    }

    public static String byQPA(int qpa) {
        if (qpa <= 3) {
            return switch (qpa) {
                case 0 -> "未用";
                case 1 -> "激活/停止激活这之前装载的参数(信息对象地址=0)";
                case 2 -> "激活/停止激活所寻址信息对象的参数";
                case 3 -> "激活/停止激活所寻址的持续循环或周期传输的信息对象";
                default -> throw new IllegalStateException("Unexpected value: " + qpa);
            };
        } else if (qpa <= 31) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("QPA=").append(byQPA(qpa));
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

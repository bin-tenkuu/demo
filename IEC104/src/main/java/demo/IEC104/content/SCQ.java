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
public class SCQ implements BaseContent {
    private int ui;
    private int bs;

    public SCQ(byte b) {
        ui = b & 0b1111;
        bs = b >>> 4;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = (byte) ((ui & 0b1111) | (bs << 4));
    }

    public static String byUI(int ui) {
        if (ui <= 7) {
            return switch (ui) {
                case 0 -> "缺省";
                case 1 -> "选择文件";
                case 2 -> "请求文件";
                case 3 -> "停止激活文件";
                case 4 -> "删除文件";
                case 5 -> "选择节";
                case 6 -> "请求节";
                case 7 -> "停止激活节";
                default -> throw new IllegalStateException("Unexpected value: " + ui);
            };
        } else if (ui <= 10) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    public static String byBS(int bs) {
        if (bs <= 5) {
            return switch (bs) {
                case 0 -> "缺省";
                case 1 -> "无所请求的存储空间";
                case 2 -> "校验和错";
                case 3 -> "非所期望的通信服务";
                case 4 -> "非所期望的文件名称";
                case 5 -> "非所期望的节名称";
                default -> throw new IllegalStateException("Unexpected value: " + bs);
            };
        } else if (bs <= 10) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("UI=").append(byUI(ui))
                .append("，BS=").append(byBS(bs))
        ;
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

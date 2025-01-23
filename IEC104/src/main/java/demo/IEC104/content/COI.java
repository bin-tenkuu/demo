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
public class COI implements BaseContent {
    private int ui;
    private boolean bsi;

    public COI(byte b) {
        ui = b & 0b0111_1111;
        bsi = ByteUtil.getBit(b, 7);
    }

    @Override
    public int size() {
        return 1;
    }

    public static String byUI(int qu) {
        if (qu <= 2) {
            return switch (qu) {
                case 0 -> "当地电源合上";
                case 1 -> "当地手动复位";
                case 2 -> "远方复位";
                default -> throw new IllegalStateException("Unexpected value: " + qu);
            };
        } else if (qu <= 31) {
            return "为本配套标准的标准定义保留(兼容范围)";
        } else {
            return "为特定使用保留(专用范围)";
        }
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = (byte) (ui & 0b0111_1111);
        b = ByteUtil.setBit(b, 7, bsi);
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("改变当地参数的初始化(BSI)=").append(bsi ? "否" : "是")
                .append("，UI=").append(byUI(ui))
        ;
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

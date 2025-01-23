package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/01/23
 */
@Getter
@Setter
public class SOF implements BaseContent {
    private boolean fa;
    private boolean fr;

    public SOF(byte b) {
        fa = ByteUtil.getBit(b, 7);
        fr = ByteUtil.getBit(b, 6);
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = 0;
        b = ByteUtil.setBit(b, 7, fa);
        b = ByteUtil.setBit(b, 6, fr);
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("文件传输(FA)=").append(fa ? "激活" : "等待")
                .append(",定义(FOR)=").append(fr ? "子目录名" : "文件名");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/13
 */
@Getter
@Setter
public class QDS extends BaseQds implements BaseContent {
    protected boolean ov;

    public QDS(byte b) {
        super(b);
        ov = ByteUtil.getBit(b, 0);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] = ByteUtil.setBit(data[offset], 0, ov);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append("，OV=").append(ov ? "已溢出" : "未溢出");
    }
}

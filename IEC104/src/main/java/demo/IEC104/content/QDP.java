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
public class QDP extends BaseQds implements BaseContent {
    private boolean ei;

    public QDP(byte b) {
        super(b);
        ei = ByteUtil.getBit(b, 3);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] = ByteUtil.setBit((data[offset]), 3, ei);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append("，EI=").append(ei ? "动作时间无效" : "动作时间有效");
    }
}

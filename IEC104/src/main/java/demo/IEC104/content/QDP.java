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
    private int event;

    public QDP(byte b) {
        super(b);
        ei = ByteUtil.getBit(b, 3);
        event = b & 3;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset] = ByteUtil.setBit((byte) (data[offset] | (event & 3)), 3, ei);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append("，EI=").append(ei ? "无效" : "有效")
                .append("，事件=").append(event);
    }
}

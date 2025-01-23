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
public class SCO extends QOC implements BaseContent {
    private boolean scs;

    public SCO(byte b) {
        super(b);
        scs = ByteUtil.getBit(b, 0);
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = ByteUtil.setBit(data[offset], 7, scs);
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append(", 单命令状态=").append(scs ? "合" : "开");
    }

}

package demo.IEC104.content;

import demo.IEC104.ByteUtil;
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
public abstract class BaseQds implements BaseContent {
    protected boolean iv;
    protected boolean nt;
    protected boolean sb;
    protected boolean bl;

    public BaseQds(byte b) {
        iv = ByteUtil.getBit(b, 7);
        nt = ByteUtil.getBit(b, 6);
        sb = ByteUtil.getBit(b, 5);
        bl = ByteUtil.getBit(b, 4);
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        byte b = 0;
        b = ByteUtil.setBit(b, 7, iv);
        b = ByteUtil.setBit(b, 6, nt);
        b = ByteUtil.setBit(b, 5, sb);
        b = ByteUtil.setBit(b, 4, bl);
        data[offset] = b;
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("IV=").append(iv ? "无效" : "有效")
                .append("，NT=").append(nt ? "非当前值" : "是当前值")
                .append("，SB=").append(sb ? "已取代" : "未取代")
                .append("，BL=").append(bl ? "已封锁" : "未封锁");
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

import java.time.LocalDateTime;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/12
 */
@Getter
@Setter
public class CP24Time2a extends CP16Time2a implements BaseContent {
    protected boolean iv;
    protected int minute;

    public CP24Time2a(byte[] content, int offset) {
        super(content, offset);
        val b2 = content[offset + 2];
        iv = ByteUtil.getBit(b2, 7);
        minute = b2 & 0b00111111;
    }

    @Override
    public int size() {
        return 3;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset + 2] = ByteUtil.setBit((byte) (minute & 0b00111111), 7, iv);
    }

    @Override
    public LocalDateTime toLocalDateTime() {
        return LocalDateTime.of(
                0,
                1,
                1,
                0,
                minute,
                millisecond / 1000,
                (millisecond % 1000) * 1000000
        );
    }

    @Override
    public void toString(StringBuilder builder) {
        super.toString(builder);
        builder.append("，IV=").append(iv ? "无效" : "有效");
    }
}

package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.val;

import java.time.LocalDateTime;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/12
 */
public class CP32Time2a extends CP24Time2a implements BaseContent {
    protected int hour;
    protected boolean su;

    public CP32Time2a(byte[] content, int offset) {
        super(content, offset);
        val b3 = content[offset + 3];
        hour = b3 & 0b00011111;
        su = ByteUtil.getBit(b3, 7);
    }

    @Override
    public int size() {
        return 4;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset + 3] = ByteUtil.setBit((byte) (hour & 0b00011111), 7, su);
    }

    @Override
    public LocalDateTime toLocalDateTime() {
        return LocalDateTime.of(
                0,
                1,
                1,
                hour,
                minute,
                millisecond / 1000,
                (millisecond % 1000) * 1000000
        );
    }
}

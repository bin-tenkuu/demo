package demo.IEC104.content;

import demo.IEC104.ByteUtil;

import java.time.LocalTime;

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
        var b3 = content[offset + 3];
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
    public LocalTime toLocalTime() {
        return LocalTime.of(
                hour,
                minute,
                millisecond / 1000,
                (millisecond % 1000) * 1000000
        );
    }
}

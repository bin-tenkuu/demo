package demo.IEC104.content;

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
public class CP56Time2a extends CP32Time2a implements BaseContent {
    protected int day;
    protected int week;
    protected int month;
    protected int year;

    public CP56Time2a(byte[] content, int offset) {
        super(content, offset);
        val b4 = content[offset + 4];
        year = content[offset + 6] & 0b01111111;
        month = content[offset + 5] & 0b00001111;
        day = b4 & 0b00011111;
        week = (b4 & 0xFF) >>> 5;
    }

    @Override
    public int size() {
        return 7;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        super.writeTo(data, offset);
        data[offset + 4] = (byte) ((week << 5) | (day & 0b00011111));
        data[offset + 5] = (byte) (month & 0b00001111);
        data[offset + 6] = (byte) (year & 0b01111111);
    }

    @Override
    public LocalDateTime toLocalDateTime() {
        return LocalDateTime.of(
                year,
                month,
                day,
                hour,
                minute,
                millisecond / 1000,
                (millisecond % 1000) * 1000000
        );
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("时间=").append(toLocalDateTime().toString())
                .append("，星期=").append(week)
                .append("，是否无效(IV)=").append(iv);
    }
}

package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/12
 */
@Getter
@Setter
public class CP16Time2a implements BaseContent {
    protected int millisecond;

    public CP16Time2a(byte[] content, int offset) {
        millisecond = ByteUtil.getShort(content, offset);
    }

    @Override
    public int size() {
        return 2;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        ByteUtil.setShort(data, offset, (short) millisecond);
    }

    public LocalDateTime toLocalDateTime() {
        return LocalDateTime.of(
                0,
                1,
                1,
                0,
                0,
                millisecond / 1000,
                (millisecond % 1000) * 1000000
        );
    }

    @Override
    public String toString() {
        return "时间=" + toLocalDateTime().toLocalTime().toString();
    }
}

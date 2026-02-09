package demo.IEC104.content;

import demo.IEC104.ByteUtil;
import lombok.Getter;
import lombok.Setter;
import lombok.val;

import java.time.LocalTime;

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

    public LocalTime toLocalTime() {
        return LocalTime.of(
                0,
                0,
                millisecond / 1000,
                (millisecond % 1000) * 1000000
        );
    }

    @Override
    public void toString(StringBuilder builder) {
        builder.append("时间=").append(toLocalTime());
    }

    @Override
    public String toString() {
        val sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }
}

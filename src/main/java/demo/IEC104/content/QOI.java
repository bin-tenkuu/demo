package demo.IEC104.content;

import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/13
 */
@Getter
@Setter
public class QOI implements BaseContent {
    private byte value;

    public QOI(byte value) {
        this.value = value;
    }

    @Override
    public int size() {
        return 1;
    }

    @Override
    public void writeTo(byte[] data, int offset) {
        data[offset] = value;
    }

    @Override
    public String toString() {
        return "QOI=" + value;
    }
}

package demo.IEC104;

import lombok.Getter;

/**
 * @author bin
 * @version 1.0.0
 * @see <a href="https://blog.redisant.cn/docs/iec104-tutorial/">IEC 104</a>
 * @since 2024/09/10
 */
@Getter
public class Frame {
    public final FrameType type;
    public final byte[] data;

    public Frame(FrameType type, byte[] data) {
        this.type = type;
        data[0] = 0x68;
        this.data = data;
    }

    public int getLength() {
        return data[1] & 0xFF;
    }

    public void setLength(int length) {
        data[1] = (byte) length;
    }

    public byte[] toByteArray() {
        return data;
    }
}

package demo.IEC104.content;

import lombok.val;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/12
 */
public interface BaseContent {

    int size();

    default byte[] toByteArray() {
        val data = new byte[size()];
        writeTo(data, 0);
        return data;
    }

    void writeTo(byte[] data, int offset);
}

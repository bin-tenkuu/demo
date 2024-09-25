package demo.IEC104;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
public class FrameS extends Frame {
    public FrameS(byte[] data) {
        super(FrameType.S, data);
    }

    public FrameS() {
        super(FrameType.S, new byte[6]);
        data[2] = 1;
    }

    // region APCI
    public int getReceiveCounte() {
        return ByteUtil.getShort(data, 4) >> 1;
    }

    public void setReceiveCounte(int receiveCounter) {
        ByteUtil.setShort(data, 4, (short) (receiveCounter << 1));
    }
    // endregion

}

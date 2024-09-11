package demo.IEC104;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
public class FrameS extends Frame {
    public FrameS(FrameType type, byte[] data) {
        super(type, data);
    }

    public FrameS() {
        super(FrameType.S, new byte[6]);
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

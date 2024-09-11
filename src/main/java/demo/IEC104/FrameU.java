package demo.IEC104;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
public class FrameU extends Frame {
    public FrameU(FrameType type, byte[] data) {
        super(type, data);
    }

    public FrameU() {
        super(FrameType.U, new byte[6]);
    }

    // region APCI

    /**
     * 0 - 不是测试帧
     * 1 - 测试
     * 2 - 测试确认
     */
    public int getTest() {
        return (data[2] >>> 6);
    }

    public void setTest(int test) {
        data[2] = (byte) (3 | (test << 6));
    }

    /**
     * 0 - 不是停止帧
     * 1 - 停止
     * 2 - 停止确认
     */
    public int getStop() {
        return (data[2] >>> 4) & 3;
    }

    public void setStop(int stop) {
        data[2] = (byte) (3 | (stop << 4));
    }

    /**
     * 0 - 不是启动帧
     * 1 - 启动
     * 2 - 启动确认
     */
    public int getStart() {
        return (data[2] >>> 2) & 3;
    }

    public void setStart(int start) {
        data[2] = (byte) (3 | (start << 2));
    }
    // endregion
}

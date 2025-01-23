package demo.IEC104;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/10
 */
public enum FrameType {
    UNKNOWN,
    /**
     * 编号的信息传输
     */
    I,
    /**
     * 编号的监视功能
     */
    S,
    /**
     * 未编号的控制功能
     */
    U
}

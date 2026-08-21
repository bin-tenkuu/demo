package demo.util;

/**
 * 模式：熔断器
 * <p>
 * 通过跟踪错误次数自动跳闸——快速失败，而不是堆积超时等待。
 *
 * @author bin
 * @since 2026/08/21
 */
@SuppressWarnings("unused")
public class CircuitBreaker {
    private static final Boolean CLOSED = null;
    private static final Boolean OPEN = Boolean.TRUE;
    private static final Boolean HALF_OPEN = Boolean.FALSE;

    private enum State {
        CLOSED, OPEN, HALF_OPEN
    }

    private static class CircuitOpenException extends RuntimeException {

    }

    public interface SupplierWithException<T> {
        T get() throws Exception;
    }

    /// 失败次数阈值，超过该阈值后熔断器进入 OPEN 状态
    private final int threshold;
    /// 重置超时时间，单位毫秒，熔断器在 OPEN 状态下等待该时间后进入 HALF_OPEN 状态
    private final int resetTimeout;

    /// 当前熔断器状态，CLOSED 表示正常，OPEN 表示熔断，HALF_OPEN 表示半开
    private Boolean state = CLOSED;
    /// 当前失败次数，超过阈值后熔断器进入 OPEN 状态
    private int failureCount = 0;
    /// 重置时间，单位毫秒，熔断器在 OPEN 状态下等待该时间后进入 HALF_OPEN 状态
    private long resetTime = 0;

    public CircuitBreaker(int threshold, int resetTimeout) {
        this.threshold = threshold;
        this.resetTimeout = resetTimeout;
    }

    public <T> T execute(SupplierWithException<T> supplier) throws Exception {
        if (OPEN.equals(state)) {
            if (System.currentTimeMillis() >= resetTime) {
                state = HALF_OPEN;
            } else {
                throw new CircuitOpenException();
            }
        }

        try {
            T result = supplier.get();
            if (HALF_OPEN.equals(state)) {
                state = CLOSED;
                failureCount = 0;
            }
            return result;
        } catch (Exception e) {
            failureCount++;
            if (failureCount >= threshold) {
                state = OPEN;
                resetTime = System.currentTimeMillis() + resetTimeout;
            }
            throw e;
        }
    }
}

package demo.threadpool;

import lombok.Setter;
import lombok.val;

import java.util.Collection;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * 优先使用最大线程数，然后使用队列长度
 *
 * @author bin
 * @version 1.0.0
 * @since 2024/07/04
 */
@SuppressWarnings("unused")
@Setter
public class TaskQueue extends LinkedBlockingQueue<Runnable> {
    private transient volatile ThreadPoolExecutor parent = null;

    public TaskQueue() {
    }

    public TaskQueue(int capacity) {
        super(capacity);
    }

    public TaskQueue(Collection<? extends Runnable> c) {
        super(c);
    }

    public boolean offer(Runnable o) {
        if (this.parent == null) {
            return super.offer(o);
        } else {
            val poolSize = this.parent.getActiveCount();
            if (poolSize >= this.parent.getMaximumPoolSize()) {
                return super.offer(o);
            } else if (this.size() <= poolSize) {
                return super.offer(o);
            } else {
                return false;
            }
        }
    }
}

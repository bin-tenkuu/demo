package test;

import demo.fsm.StateMachineLambda;
import lombok.RequiredArgsConstructor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Function;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/04/07
 */
@DisplayName("状态机测试")
public class StateMachineLambdaTest {
    @RequiredArgsConstructor
    private static class IntArray {
        private final int[] arr;
        private int index;

        public boolean hasNext() {
            return index < arr.length;
        }

        public int view() {
            if (index >= arr.length) {
                throw new IllegalStateException("No more elements");
            }
            return arr[index];
        }

        public void next() {
            index++;
        }
    }

    private record Stats(int state, List<Integer> nexts) implements Function<IntArray, List<Integer>> {
        @Override
        public List<Integer> apply(IntArray arr) {
            if (arr.hasNext()) {
                if (arr.view() == state) {
                    arr.next();
                    System.out.print(state);
                    return nexts;
                }
            }
            return null;
        }
    }

    private StateMachineLambda<Integer, IntArray> lambda;

    @BeforeEach
    public void setUp() {
        // 1-2
        // 2-3,4
        // 3-1,5
        // 4-
        lambda = new StateMachineLambda<>();
        lambda.add(1, new Stats(1, List.of(2)));
        lambda.add(2, new Stats(2, List.of(3, 4)));
        lambda.add(3, new Stats(3, List.of(1, 5)));
        lambda.add(4, new Stats(4, List.of()));
    }

    @Test
    @DisplayName("顺序")
    public void t1() {
        lambda.execute(1, new IntArray(new int[]{
                1, 2, 3, 1, 2, 4
        }));
    }

    @Test
    @DisplayName("未匹配")
    public void t2() {
        lambda.execute(1, new IntArray(new int[]{
                1, 3
        }));
    }

    @Test
    @DisplayName("未预期结束")
    public void t3() {
        lambda.setContinueInUnStateEnd(false);
        try {
            lambda.execute(3, new IntArray(new int[]{
                    3, 1
            }));
        } catch (Exception e) {
            System.out.println("异常: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("未知的状态")
    public void t4() {
        lambda.setContinueInNoAction(false);
        try {
            lambda.execute(3, new IntArray(new int[]{
                    3, 5
            }));
        } catch (Exception e) {
            System.out.println("异常: " + e.getMessage());
        }
    }

}

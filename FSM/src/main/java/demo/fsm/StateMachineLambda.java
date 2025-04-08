package demo.fsm;

import lombok.Setter;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.function.Function;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/04/07
 */
@Setter
public class StateMachineLambda<K, E> {
    private final HashMap<K, Function<E, ? extends Collection<? extends K>>> actionMap = new HashMap<>();
    private boolean continueInNoAction = false;
    private boolean continueInUnStateEnd = true;

    /**
     * @param state 状态名
     * @param action 返回null时表示无法处理，返回空集合表示结束，返回非空集合表示按顺序判断处理
     */
    public void add(K state, Function<E, ? extends Collection<? extends K>> action) {
        actionMap.put(state, action);
    }

    public void remove(K state) {
        actionMap.remove(state);
    }

    public void clear() {
        actionMap.clear();
    }

    public void execute(K startState, E event) {
        execute(Collections.singleton(startState), event);
    }

    public void execute(Collection<? extends K> startStates, E event) {
        var states = startStates;
        outter:
        while (true) {
            for (var state : states) {
                var action = actionMap.get(state);
                if (action == null) {
                    if (continueInNoAction) {
                        continue;
                    } else {
                        throw new IllegalStateException("State " + state + " not found in action map");
                    }
                }
                var tmp = action.apply(event);
                if (tmp == null) {
                    continue;
                }
                if (tmp.isEmpty()) {
                    return;
                }
                states = tmp;
                continue outter;
            }
            if (continueInUnStateEnd) {
                return;
            } else {
                throw new IllegalStateException("No valid state found in action map");
            }
        }
    }
}

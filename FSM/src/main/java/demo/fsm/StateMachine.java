package demo.fsm;

import lombok.Setter;
import lombok.val;

import java.util.HashMap;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * 基于Lambda的泛型状态机
 *
 * @param <S> 状态类型
 * @param <E> 事件类型
 * @author bin
 * @version 1.0.0
 * @since 2025/04/07
 */
@SuppressWarnings("unused")
@Setter
public class StateMachine<S, E> {
    /**
     * 状态转移规则：状态 -> 转移函数
     */
    private final HashMap<S, BiFunction<S, E, S>> transitions = new HashMap<>();
    /**
     * 默认转移函数
     */
    private BiFunction<S, E, S> defaultTransition;
    /**
     * 状态变化监听器
     */
    private StateChangeListener<S, E> stateChangeListener;
    /**
     * 当前状态
     */
    private S currentState;

    /**
     * 状态变化监听器接口
     */
    @FunctionalInterface
    public interface StateChangeListener<S, E> {
        void onStateChanged(S fromState, S toState, E event);
    }

    public StateMachine() {
        this.defaultTransition = (s, e) -> s;
    }

    public StateMachine(BiFunction<S, E, S> defaultTransition) {
        this.defaultTransition = Objects.requireNonNull(defaultTransition, "默认转移函数不能为空");
    }

    /**
     * 注册状态转移规则
     *
     * @param state 状态
     * @param transition 转移函数 (当前状态, 事件) -> 下一状态
     */
    public void registerState(S state, BiFunction<S, E, S> transition) {
        transitions.put(state, transition);
    }

    public void registerState(S state, Function<E, S> transition) {
        transitions.put(state, (_, e) -> transition.apply(e));
    }

    public void registerState(S state, Supplier<S> transition) {
        transitions.put(state, (_, _) -> transition.get());
    }

    public void remove(S sourceState) {
        transitions.remove(sourceState);
    }

    public void clear() {
        transitions.clear();
    }

    public S handleEvent(E event) {
        return currentState = handleEvent(currentState, event);
    }

    /**
     * 处理事件并执行状态转移
     *
     * @param event 触发事件
     */
    public S handleEvent(S currentState, E event) {
        BiFunction<S, E, S> transition = transitions.getOrDefault(currentState, defaultTransition);

        val newState = transition.apply(currentState, event);

        if (stateChangeListener != null) {
            stateChangeListener.onStateChanged(currentState, currentState, event);
        }
        return newState;
    }

}

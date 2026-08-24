package demo.util;

import java.util.HashMap;
import java.util.Map;

/**
 * @author bin
 * @since 2026/08/24
 */
public class StateMachine<State, Event> {
    private final Map<State, Map<Event, State>> config = new HashMap<>();
    private State current;

    public StateMachine(Map<State, Map<Event, State>> config, State initialState) {
        this.config.putAll(config);
        this.current = initialState;
    }

    public StateMachine(State initialState) {
        this.current = initialState;
    }

    public void addTransition(State from, Event event, State to) {
        config.computeIfAbsent(from, _ -> new HashMap<>()).put(event, to);
    }

    public void addTransition(State from, Map<Event, State> transitions) {
        config.computeIfAbsent(from, _ -> new HashMap<>()).putAll(transitions);
    }

    public State getState() {
        return current;
    }

    public State sendEvent(Event event) {
        var transitions = config.get(current);
        if (transitions != null) {
            var nextState = transitions.get(event);
            if (nextState != null) {
                current = nextState;
            }
        }
        return current;
    }

    public boolean canSendEvent(Event event) {
        var transitions = config.get(current);
        if (transitions == null) {
            return false;
        }
        return transitions.containsKey(event);
    }

    static void main() {
        enum State {
            IDLE, LOADING, SUCCESS, ERROR
        }
        enum Event {
            FETCH, RESOLVE, REJECT, RETRY, RESET
        }

        var machine = new StateMachine<State, Event>(State.IDLE);
        machine.addTransition(State.IDLE, Event.FETCH, State.LOADING);
        machine.addTransition(State.LOADING, Map.of(
                Event.RESOLVE, State.SUCCESS,
                Event.REJECT, State.ERROR
        ));
        machine.addTransition(State.ERROR, Event.RETRY, State.LOADING);
        machine.addTransition(State.SUCCESS, Event.RESET, State.IDLE);
        machine.sendEvent(Event.FETCH);
        System.out.println(machine.getState()); // LOADING
        machine.sendEvent(Event.REJECT);
        System.out.println(machine.getState()); // ERROR
        machine.sendEvent(Event.RETRY);
        System.out.println(machine.getState()); // LOADING
        machine.sendEvent(Event.RESOLVE);
        System.out.println(machine.getState()); // SUCCESS
        machine.sendEvent(Event.RESET);
        System.out.println(machine.getState()); // IDLE
    }
}

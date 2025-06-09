package test;

import demo.fsm.StateMachine;
import lombok.RequiredArgsConstructor;
import lombok.val;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.function.IntUnaryOperator;

/**
 * @author bin
 * @version 1.0.0
 * @since 2025/04/07
 */
@DisplayName("状态机测试")
public class StateMachineTest {
    @Test
    @DisplayName("基础状态机测试")
    public void homeDevice() {
        val homeDevice = new StateMachine<String, String>();
        // 设置状态变化监听器
        homeDevice.setStateChangeListener((from, to, event) ->
                System.out.printf("[状态变化] %s -> %s (事件: %s)%n", from, to, event)
        );

        // 注册各状态的转移规则
        homeDevice.registerState("OFF", (state, event) -> switch (event) {
            case "power" -> {
                System.out.println("设备开机");
                yield "ON";
            }
            case null, default -> state;
        });

        homeDevice.registerState("ON", (state, event) -> switch (event) {
            case "power" -> {
                System.out.println("设备关机");
                yield "OFF";
            }
            case "standby" -> {
                System.out.println("进入待机模式");
                yield "STANDBY";
            }
            case "update" -> {
                System.out.println("开始更新固件");
                yield "UPDATING";
            }
            case "error" -> {
                System.out.println("发生错误");
                yield "ERROR";
            }
            default -> state;
        });

        homeDevice.registerState("STANDBY", (state, event) -> switch (event) {
            case "power" -> {
                System.out.println("从待机状态关机");
                yield "OFF";
            }
            case "wake" -> {
                System.out.println("从待机状态唤醒");
                yield "ON";
            }
            case null, default -> state;
        });

        homeDevice.registerState("UPDATING", (state, event) -> switch (event) {
            case "complete" -> {
                System.out.println("更新完成");
                yield "ON";
            }
            case "error" -> {
                System.out.println("更新失败");
                yield "ERROR";
            }
            case "cancel" -> {
                System.out.println("取消更新");
                yield "ON";
            }
            case null, default -> state;
        });

        homeDevice.registerState("ERROR", (state, event) -> switch (event) {
            case "reset" -> {
                System.out.println("设备重置");
                yield "OFF";
            }
            case "retry" -> {
                System.out.println("尝试恢复操作");
                yield "ON";
            }
            case null, default -> state;
        });
        homeDevice.setCurrentState("OFF"); // 初始状态为 OFF
        homeDevice.handleEvent("power"); // OFF -> ON
        homeDevice.handleEvent("standby"); // ON -> STANDBY
        homeDevice.handleEvent("wake"); // STANDBY -> ON
        homeDevice.handleEvent("update"); // ON -> UPDATING
        homeDevice.handleEvent("error"); // UPDATING -> ERROR
        homeDevice.handleEvent("retry"); // ERROR -> ON
        homeDevice.handleEvent("power"); // ON -> OFF

        // 测试无效事件
        System.out.println("\n测试无效事件:");
        homeDevice.handleEvent("invalid"); // OFF -> OFF (保持)
    }

    @Test
    @DisplayName("计算器状态机测试")
    public void mathCalc() {

        val mathMachine = new StateMachine<String, MathState>((_, _) -> {
            throw new UnsupportedOperationException("未定义的操作符");
        });
        mathMachine.registerState(null, _ -> "+");
        mathMachine.registerState("", e -> {
            e.number = e.nextNumber();
            return e.nextOperator();
        });

    }

    @RequiredArgsConstructor
    static class MathState {
        private final String expression;
        private int index = 0;
        private int number = 0;
        private IntUnaryOperator intHandler;

        public boolean hasNext() {
            return index < expression.length();
        }

        public void setNumber(int number) {

        }

        public int nextNumber() {
            int start = index;
            while (index < expression.length() && Character.isDigit(expression.charAt(index))) {
                index++;
            }
            return Integer.parseInt(expression.substring(start, index));
        }

        public String nextOperator() {
            int start = index;
            while (index < expression.length() && !Character.isDigit(expression.charAt(index))) {
                index++;
            }
            return expression.substring(start, index);
        }
    }
}

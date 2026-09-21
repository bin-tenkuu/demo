package demo.apt;

import java.lang.annotation.*;

/**
 * @author bin
 * @since 2026/09/21
 */
@Documented
@Target({ElementType.TYPE, ElementType.PACKAGE})
@Retention(RetentionPolicy.SOURCE)
public @interface AutoDemo {
    /// 设置为 true 时跳过处理
    boolean value() default false;
}

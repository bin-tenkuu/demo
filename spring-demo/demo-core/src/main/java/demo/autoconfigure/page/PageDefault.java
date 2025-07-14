package demo.autoconfigure.page;

import java.lang.annotation.*;

/**
 * Annotation to set defaults when injecting a {@link com.baomidou.mybatisplus.core.metadata.IPage} into a controller
 * method.
 *
 * @author bin
 * @since 2022/12/16
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.PARAMETER)
public @interface PageDefault {
    /**
     * Alias for {@link #size()}. Prefer to use the {@link #size()} method as it makes the annotation declaration more
     * expressive and you'll probably want to configure the {@link #page()} anyway.
     *
     * @return page
     */
    int value() default 10;

    /**
     * The default-size the injected {@link com.baomidou.mybatisplus.core.metadata.IPage} should get if no corresponding
     * parameter defined in request (default is 5).
     *
     * @return size
     */
    int size() default 5;

    /**
     * The default-pagenumber the injected {@link com.baomidou.mybatisplus.core.metadata.IPage} should get if no corresponding
     * parameter defined in request (default is 0).
     *
     * @return page
     */
    int page() default 0;

}

package demo.autoconfigure.page;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import org.jetbrains.annotations.NotNull;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.core.MethodParameter;
import org.springframework.lang.Nullable;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.support.WebDataBinderFactory;
import org.springframework.web.context.request.NativeWebRequest;
import org.springframework.web.method.support.HandlerMethodArgumentResolver;
import org.springframework.web.method.support.ModelAndViewContainer;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.util.List;
import java.util.OptionalLong;

/**
 * @author bin
 * @since 2023/01/15
 */
@EnableConfigurationProperties(MybatisPageProperties.class)
public class MybatisPageWebMvcConfigurer implements HandlerMethodArgumentResolver, WebMvcConfigurer {
    private final MybatisPageProperties properties;

    public MybatisPageWebMvcConfigurer(final MybatisPageProperties properties) {
        this.properties = properties;
    }

    @Override
    public boolean supportsParameter(MethodParameter parameter) {
        return IPage.class.isAssignableFrom(parameter.getParameterType());
    }

    @NotNull
    protected IPage<?> getDefaultFromAnnotationOrFallback(@NotNull MethodParameter parameter) {
        final PageDefault pageDefault = parameter.getParameterAnnotation(PageDefault.class);
        if (pageDefault != null) {
            int page = pageDefault.page();
            if (page < 1) {
                page = pageDefault.value();
            }
            return Page.of(page, pageDefault.size());
        }
        return Page.of(1, properties.getDefaultPageSize());
    }

    @NotNull
    protected IPage<?> getPage(
            @NotNull MethodParameter parameter, @Nullable String pageString, @Nullable String pageSizeString
    ) {
        final IPage<?> fallback = getDefaultFromAnnotationOrFallback(parameter);

        final OptionalLong pageOptional = parseAndApplyBoundaries(pageString, Long.MAX_VALUE);
        final OptionalLong pageSizeOptional = parseAndApplyBoundaries(pageSizeString, properties.getMaxPageSize());

        if (!(pageOptional.isPresent() || pageSizeOptional.isPresent())) {
            return fallback;
        }
        final long page = pageOptional.orElse(fallback.getCurrent());
        final long size = pageSizeOptional.orElse(fallback.getSize());

        return Page.of(page, size);

    }

    @Override
    public IPage<?> resolveArgument(
            @NotNull MethodParameter parameter, ModelAndViewContainer mavContainer,
            @NotNull NativeWebRequest webRequest, WebDataBinderFactory binderFactory
    ) {
        final String pageString = webRequest.getParameter(properties.getPageParameter());
        final String pageSizeString = webRequest.getParameter(properties.getSizeParameter());

        return getPage(parameter, pageString, pageSizeString);
    }

    private static OptionalLong parseAndApplyBoundaries(@Nullable String parameter, long upper) {
        if (!StringUtils.hasText(parameter)) {
            return OptionalLong.empty();
        } else {
            try {
                final long parsed = Long.parseLong(parameter, 10);
                return OptionalLong.of(parsed < 0 ? 0 : Math.min(parsed, upper));
            } catch (NumberFormatException var5) {
                return OptionalLong.of(0L);
            }
        }
    }

    @Override
    public void addArgumentResolvers(final List<HandlerMethodArgumentResolver> resolvers) {
        resolvers.add(this);
    }

}

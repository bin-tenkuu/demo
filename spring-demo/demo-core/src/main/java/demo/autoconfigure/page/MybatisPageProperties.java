package demo.autoconfigure.page;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration properties for Spring Data Web.
 *
 * @author bin
 * @since 2022/11/29
 */
@Setter
@Getter
@ConfigurationProperties("spring.data.web.pageable")
public class MybatisPageProperties {
    private static final String DEFAULT_PAGE_PARAMETER = "page";
    private static final String DEFAULT_SIZE_PARAMETER = "size";
    private static final int DEFAULT_PAGE_SIZE = 5;
    private static final int DEFAULT_MAX_PAGE_SIZE = 2000;

    /**
     * Page index parameter name.
     */
    private String pageParameter = DEFAULT_PAGE_PARAMETER;

    /**
     * Page size parameter name.
     */
    private String sizeParameter = DEFAULT_SIZE_PARAMETER;

    /**
     * Default page size.
     */
    private long defaultPageSize = DEFAULT_PAGE_SIZE;

    /**
     * Maximum page size to be accepted.
     */
    private long maxPageSize = DEFAULT_MAX_PAGE_SIZE;

}

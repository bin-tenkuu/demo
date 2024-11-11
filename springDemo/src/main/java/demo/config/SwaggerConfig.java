package demo.config;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeIn;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import io.swagger.v3.oas.annotations.security.SecuritySchemes;
import io.swagger.v3.oas.models.media.Schema;
import org.springdoc.core.utils.SpringDocUtils;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

/**
 * @author 杨启东
 * @since 2023/03/10
 */
@OpenAPIDefinition(
        info = @Info(
                title = "综合能源边缘管理系统 接口",
                description = "综合能源边缘管理系统 相关 API",
                version = "v1.0"
        ),
        security = {
                @SecurityRequirement(name = "Authorization")
        }
)
@SecuritySchemes({
        @SecurityScheme(
                type = SecuritySchemeType.HTTP,
                name = "Authorization",
                in = SecuritySchemeIn.HEADER,
                scheme = "bearer",
                bearerFormat = "Bearer "
        )
})
@Component
public class SwaggerConfig {
    static {
        SpringDocUtils.getConfig()
                .replaceWithSchema(LocalDate.class, new Schema<>()
                        .type("string")
                        .description("yyyy-MM-dd")
                        .example("2020-01-01")
                )
                .replaceWithSchema(LocalTime.class, new Schema<>()
                        .type("string")
                        .description("HH:mm:ss")
                        .example("01:02:03")
                )
                .replaceWithSchema(LocalDateTime.class, new Schema<>()
                        .type("string")
                        .description("yyyy-MM-dd HH:mm:ss")
                        .example("2020-01-01 01:02:03")
                )
        ;
    }
}

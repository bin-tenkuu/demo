package demo.autoconfigure.swagger;

import com.baomidou.mybatisplus.core.metadata.IPage;
import demo.constant.DateConstant;
import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeIn;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import io.swagger.v3.oas.annotations.security.SecuritySchemes;
import io.swagger.v3.oas.models.media.Schema;
import org.springdoc.core.utils.SpringDocUtils;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

/**
 * @author bin
 * @since 2023/03/10
 */
@OpenAPIDefinition(
        info = @Info(
                title = "接口",
                description = "相关 API",
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
public class SwaggerCommonConfigure {
    static {
        SpringDocUtils.getConfig()
                .replaceParameterObjectWithClass(IPage.class, Pageable.class)
                .replaceWithSchema(LocalDate.class, new Schema<>()
                        .type("string")
                        .description(DateConstant.DATE_FORMAT)
                        .example("2020-01-01")
                )
                .replaceWithSchema(LocalTime.class, new Schema<>()
                        .type("string")
                        .description(DateConstant.TIME_FORMAT)
                        .example("01:02:03")
                )
                .replaceWithSchema(LocalDateTime.class, new Schema<>()
                        .type("string")
                        .description(DateConstant.DATE_TIME_FORMAT)
                        .example("2020-01-01 01:02:03")
                )
        ;
    }
}

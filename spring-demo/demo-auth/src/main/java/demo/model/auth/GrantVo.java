package demo.model.auth;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

/**
 * @author bin
 * @since 2023/06/01
 */
@Getter
@Setter
@Schema(description = "授权")
public class GrantVo {
    @Schema(description = "操作的id")
    @NotNull
    private Long id;
    @Schema(description = "授权的ids")
    @NotEmpty
    private List<Long> targetIds;
}

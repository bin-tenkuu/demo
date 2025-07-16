package demo.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @since 2025/07/16
 */
@Getter
@Setter
public class RegisterModel {
    private String username;
    private String password;
    @Schema(description = "登陆类型")
    private String type;
}

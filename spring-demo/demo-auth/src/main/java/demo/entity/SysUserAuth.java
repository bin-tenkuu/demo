package demo.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Getter;
import lombok.Setter;

/**
 * @author bin
 * @since 2025/07/15
 */
@Getter
@Setter
public class SysUserAuth {
    @TableId
    private String username;
    @Schema(description = "登陆类型")
    private String type;
    private Long userId;
    private String password;
}

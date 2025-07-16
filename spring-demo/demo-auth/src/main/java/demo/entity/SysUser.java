package demo.entity;

import com.baomidou.mybatisplus.annotation.TableId;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * @author bin
 * @since 2025/07/15
 */
@Getter
@Setter
public class SysUser {
    @TableId
    private Long id;
    private String nickname;
    private String email;
    private String phoneNumber;
    // private String password;
    private LocalDateTime expireTime;
    private Boolean status;
}
